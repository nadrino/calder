import torch
import numpy as np

class EventTable:
    def __init__(self, data_dict, device="cpu"):
        self.device = torch.device(device)
        self.data = {k: torch.as_tensor(np.vstack(v), device=self.device) for k, v in data_dict.items()}

    def __getitem__(self, key):
        return self.data[key]

    def keys(self):
        return list(self.data.keys())

    def apply_mask(self, mask):
        for k in self.data:
            self.data[k] = self.data[k][mask]
        return self

    def to(self, device):
        self.device = torch.device(device)
        for k in self.data:
            self.data[k] = self.data[k].to(self.device)
        return self
