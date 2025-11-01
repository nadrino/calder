import torch
from typing import Dict, Tuple
from calder.core.globals.device import get_device


class Binning:
    def __init__(self, bins: Dict[str, Tuple[torch.Tensor, torch.Tensor]]):
        """
        bins: dict of variable_name -> (low_edges, high_edges)
        Each low_edges and high_edges is a 1D torch.Tensor of equal length.
        All tensors are moved to the current device returned by get_device().
        """
        device = get_device()
        self.device = device

        # Move all bins to device
        self.bins = {
            var: (low.to(device), high.to(device))
            for var, (low, high) in bins.items()
        }

        self.variables = list(self.bins.keys())

        # Sanity check: consistent number of bins for all variables
        n_bins = {v: len(low) for v, (low, high) in self.bins.items()}
        if len(set(n_bins.values())) != 1:
            raise ValueError(f"Inconsistent bin counts: {n_bins}")
        self.n_bins = list(n_bins.values())[0]

    def contains(self, events: Dict[str, torch.Tensor]):
        """Return boolean mask for events inside any defined bin."""
        mask = torch.ones(len(next(iter(events.values()))), dtype=torch.bool, device=self.device)
        for var, (low, high) in self.bins.items():
            x = events[var].to(self.device)
            var_mask = (x[:, None] >= low) & (x[:, None] < high)
            mask &= var_mask.any(dim=1)
        return mask

    def bin_index(self, events: Dict[str, torch.Tensor]):
        """
        Return per-event integer indices for each variable.
        (N_events, N_variables)
        """
        indices = []
        for var, (low, high) in self.bins.items():
            x = events[var].to(self.device)
            idx = torch.sum(x[:, None] >= low, dim=1) - 1
            indices.append(idx)
        return torch.stack(indices, dim=1).to(self.device)
