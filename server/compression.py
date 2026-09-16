import torch


def quantize_state_dict_bytes(state_dict, dtype=torch.float16):
    """Estimate bytes after quantizing tensors to given dtype.

    Returns total bytes (assumes contiguous storage).
    """
    bytes_per_elem = 2 if dtype == torch.float16 else 4
    total = 0
    for v in state_dict.values():
        if isinstance(v, torch.Tensor):
            total += v.numel() * (torch.finfo(dtype).bits // 8)
    return total


def quantize_state_dict(state_dict, dtype=torch.float16):
    """Return a new state_dict with tensors converted to `dtype`.
    Non-tensor entries are left as-is."""
    q = {}
    for k, v in state_dict.items():
        if isinstance(v, torch.Tensor):
            q[k] = v.to(dtype)
        else:
            q[k] = v
    return q


def dequantize_state_dict(state_dict, dtype=torch.float32):
    """Return a new state_dict with tensors converted to `dtype` (default float32)."""
    dq = {}
    for k, v in state_dict.items():
        if isinstance(v, torch.Tensor):
            dq[k] = v.to(dtype)
        else:
            dq[k] = v
    return dq
