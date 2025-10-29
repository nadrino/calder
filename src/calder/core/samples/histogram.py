import torch
import math
from calder.core.globals.device import get_device

class Histogram:
    """nD histogram on CALDER global device with optional sum of weights squared."""

    def __init__(self, variables, bin_edges, weight_var="weight", track_sumw2=False):
        if len(variables) != len(bin_edges):
            raise ValueError("variables and bin_edges must have the same length")

        self.variables = list(variables)
        self.weight_var = weight_var
        self.device = get_device()
        self.track_sumw2 = track_sumw2

        # move bin edges to device, record nbins
        self.bin_edges = [edges.to(self.device) for edges in bin_edges]
        self.nbins = [len(edges) - 1 for edges in bin_edges]
        self.ndim = len(self.nbins)

        # main histogram
        self.hist = torch.zeros(self.nbins, device=self.device)
        self.sumw2 = (
            torch.zeros_like(self.hist) if self.track_sumw2 else None
        )

        # precompute strides for flattening
        strides = []
        for i in range(self.ndim):
            stride = 1
            for nb in self.nbins[i + 1 :]:
                stride *= nb
            strides.append(stride)
        self.strides = torch.tensor(strides, device=self.device, dtype=torch.long)

        self.total_bins = int(math.prod(self.nbins))
        self.strides = torch.tensor(
            [int(math.prod(self.nbins[i + 1:])) if i + 1 < len(self.nbins) else 1
             for i in range(len(self.nbins))],
            device=self.device,
            dtype=torch.long,
        )

        # detect uniform edges to avoid bucketize
        self._uniform = []
        self._lo = []
        self._invw = []
        for e in self.bin_edges:
            diffs = e[1:] - e[:-1]
            is_uniform = bool(torch.allclose(diffs, diffs[0].expand_as(diffs), rtol=0.0, atol=1e-7))
            self._uniform.append(is_uniform)
            if is_uniform:
                # store 0D tensors on device to avoid .to(...) in fill
                self._lo.append(e[0])
                # invw = nbins / (max - min)
                self._invw.append((e.numel() - 1) / (e[-1] - e[0]))
            else:
                self._lo.append(None)
                self._invw.append(None)

    def fill(self, events):
        dev = self.device

        # weights must already be on device per EventTable contract
        w = events[self.weight_var].reshape(-1)
        N = w.numel()

        # prepare flat indices and validity
        flat_index = torch.zeros(N, dtype=torch.long, device=dev)
        valid = torch.ones(N, dtype=torch.bool, device=dev)

        # compute per-dimension indices without any device moves
        for i, (var, edges) in enumerate(zip(self.variables, self.bin_edges)):
            x = events[var].reshape(-1)

            if self._uniform[i]:
                # idx = floor((x - lo) * invw)
                idx = torch.floor((x - self._lo[i]) * self._invw[i]).to(torch.long)
            else:
                # generic edges
                idx = torch.bucketize(x, edges, right=False).to(torch.long) - 1

            nb = self.nbins[i]
            # update global validity across dims
            valid &= (idx >= 0) & (idx < nb)

            # accumulate flat index contribution
            flat_index += idx * self.strides[i]

        # filter by validity once
        if not torch.any(valid):
            return self

        flat_index = flat_index[valid]
        w = w[valid]

        # reduce with bincount (often faster than scatter_add on GPU)
        hist_flat = self.hist.view(-1)
        incr = torch.bincount(flat_index, weights=w, minlength=self.total_bins)
        hist_flat.add_(incr)

        if self.track_sumw2:
            s2_flat = self.sumw2.view(-1)
            incr2 = torch.bincount(flat_index, weights=w * w, minlength=self.total_bins)
            s2_flat.add_(incr2)

        return self

    def variance(self):
        """Return variance (sumw2) on device."""
        if not self.track_sumw2:
            raise RuntimeError("sumw2 tracking is disabled for this histogram.")
        return self.sumw2
