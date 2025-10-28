# calder/core/data/histogram.py
import torch
from calder.core.globals.device import get_device

class Histogram:
    """nD histogram that bins directly on the CALDER global device."""

    def __init__(self, variables, bin_edges, weight_var="weight"):
        if len(variables) != len(bin_edges):
            raise ValueError("variables and bin_edges must have the same length")

        self.variables = list(variables)
        self.weight_var = weight_var
        self.device = get_device()

        # move bin edges to device, record nbins
        self.bin_edges = [edges.to(self.device) for edges in bin_edges]
        self.nbins = [len(edges) - 1 for edges in bin_edges]
        self.ndim = len(self.nbins)

        # main histogram
        self.hist = torch.zeros(self.nbins, device=self.device)

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

        # build a single mask for valid ranges across all dims
        mask = torch.ones(N, dtype=torch.bool, device=dev)
        for var, edges in zip(self.variables, self.bin_edges):
            x = events[var].to(dev).reshape(-1)
            mask &= (x >= edges[0]) & (x < edges[-1])

        # nothing to do if no valid event
        if mask.sum() == 0:
            return self

        # filtered weights and indices
        w = w[mask]
        flat_index = torch.zeros_like(w, dtype=torch.long)

        # compute flattened bin index
        for stride, var, edges in zip(self.strides, self.variables, self.bin_edges):
            x = events[var].to(dev).reshape(-1)[mask]
            idx = torch.bucketize(x, edges, right=False) - 1
            flat_index += idx * stride

        # fill histogram
        hist_flat = self.hist.view(-1)
        hist_flat.scatter_add_(0, flat_index, w)
        return self

    def to_cpu(self):
        """Return CPU copy of the histogram tensor."""
        return self.hist.detach().to("cpu")

    def numpy(self):
        """Return numpy array for convenience."""
        return self.to_cpu().numpy()
