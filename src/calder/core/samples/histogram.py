import torch
from calder.core.globals.device import get_device

class Histogram:
    """nD histogram on CALDER global device with optional sum of weights squared."""

    def __init__(self, variables, bin_edges, weight_var="weight", track_sumw2=True):
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

    def fill(self, events):
        dev = self.device
        w = events[self.weight_var].to(dev).reshape(-1)
        N = w.numel()

        # Build a single mask for valid ranges across all dims
        mask = torch.ones(N, dtype=torch.bool, device=dev)
        for var, edges in zip(self.variables, self.bin_edges):
            x = events[var].to(dev).reshape(-1)
            mask &= (x >= edges[0]) & (x < edges[-1])

        if mask.sum() == 0:
            return self

        # filtered weights and flat index
        w = w[mask]
        flat_index = torch.zeros_like(w, dtype=torch.long)

        for stride, var, edges in zip(self.strides, self.variables, self.bin_edges):
            x = events[var].to(dev).reshape(-1)[mask]
            idx = torch.bucketize(x, edges, right=False) - 1
            flat_index += idx * stride

        # fill main histogram
        hist_flat = self.hist.view(-1)
        hist_flat.scatter_add_(0, flat_index, w)

        # fill sum of weights squared if enabled
        if self.track_sumw2:
            sumw2_flat = self.sumw2.view(-1)
            sumw2_flat.scatter_add_(0, flat_index, w * w)

        return self

    def variance(self):
        """Return variance (sumw2) on device."""
        if not self.track_sumw2:
            raise RuntimeError("sumw2 tracking is disabled for this histogram.")
        return self.sumw2
