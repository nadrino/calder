import torch

_device = torch.device("cpu")


def set_device(device_str: str):
    """Select global device used across CALDER runtime."""
    global _device
    dev = torch.device(device_str)
    # Probe if the device is usable
    try:
        torch.empty(1, device=dev)
    except Exception as e:
        raise RuntimeError(f"Device '{device_str}' not available: {e}")
    _device = dev
    return _device


def get_device() -> torch.device:
    """Return the globally selected device."""
    return _device
