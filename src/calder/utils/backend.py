import torch

from calder.core.globals.device import get_device


def compatible_tensor(tensor: torch.Tensor) -> torch.Tensor:
    """Return a tensor cast and moved to the global device with a supported dtype."""
    dev = get_device()
    dt = tensor.dtype

    # Try the current dtype directly
    try:
        torch.empty(1, dtype=dt, device=dev)
        return tensor.to(dev)
    except Exception:
        pass

    # Try compatible fallbacks
    if tensor.is_floating_point():
        for fb in (torch.float32, torch.float16, torch.bfloat16):
            try:
                torch.empty(1, dtype=fb, device=dev)
                return tensor.to(dev, dtype=fb)
            except Exception:
                continue

    if tensor.is_complex():
        for fb in (torch.complex64,):
            try:
                torch.empty(1, dtype=fb, device=dev)
                return tensor.to(dev, dtype=fb)
            except Exception:
                continue

    if tensor.dtype == torch.int64:
        for fb in (torch.int32,):
            try:
                torch.empty(1, dtype=fb, device=dev)
                return tensor.to(dev, dtype=fb)
            except Exception:
                continue

    raise RuntimeError(f"No compatible dtype found for device {dev} and dtype {dt}")
