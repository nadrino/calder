import torch
from calder.core.globals.device import get_device
from calder.utils.backend import compatible_tensor


class EventTable:
    """Lightweight GPU/CPU container for named tensors using CALDER global device."""
    def __init__(self, data_dict: dict[str, torch.Tensor]):
        dev = get_device()
        self.data = {}
        for name, tensor in data_dict.items():
            if not isinstance(tensor, torch.Tensor):
                raise TypeError(f"Entry '{name}' is not a torch.Tensor")
            tensor = tensor.contiguous()
            self.data[name] = compatible_tensor(tensor)

    def __getitem__(self, key):
        return self.data[key]

    def keys(self):
        return list(self.data.keys())

    def apply_mask(self, mask: torch.Tensor):
        for k in self.data:
            self.data[k] = self.data[k][mask]
        return self

    def to_cpu(self) -> dict[str, torch.Tensor]:
        """Return a dict of CPU tensors for user-level access.

        - If tensors already live on CPU, returns them directly (no copy).
        - If tensors are on another device, returns new CPU copies.
        - Does not modify the EventTable in-place.
        """
        if get_device().type == "cpu":
            return self.data

        cpu_data = {}
        for k, v in self.data.items():
            # copy but keep the original intact on device
            cpu_data[k] = v.detach().to("cpu")
        return cpu_data
