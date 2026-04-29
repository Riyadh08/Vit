from contextlib import contextmanager
from typing import Dict

import torch
import torch.nn as nn


SUPPORTED_BITS = {2, 3, 4}


def quantize_tensor(x: torch.Tensor, bit: int, eps: float = 1e-8) -> torch.Tensor:
    if bit not in SUPPORTED_BITS:
        raise ValueError(f"Unsupported bit-width: {bit}")
    x_fp32 = x.detach().to(torch.float32)
    x_min = torch.min(x_fp32)
    x_max = torch.max(x_fp32)
    if torch.isclose(x_min, x_max):
        return x.detach().clone()
    qmin = 0
    qmax = (1 << bit) - 1
    scale = (x_max - x_min) / max(qmax - qmin, 1)
    scale = torch.clamp(scale, min=eps)
    zp = torch.round(qmin - x_min / scale)
    zp = torch.clamp(zp, qmin, qmax)
    q = torch.round(x_fp32 / scale + zp)
    q = torch.clamp(q, qmin, qmax)
    dq = (q - zp) * scale
    return dq.to(dtype=x.dtype, device=x.device)


@contextmanager
def temporary_quantized_modules(module_to_bit: Dict[nn.Module, int]):
    originals = {}
    with torch.no_grad():
        for module, bit in module_to_bit.items():
            originals[module] = {
                "weight": module.weight.detach().clone(),
                "bias": module.bias.detach().clone() if module.bias is not None else None,
            }
            module.weight.data.copy_(quantize_tensor(module.weight.data, bit))
            if module.bias is not None:
                module.bias.data.copy_(quantize_tensor(module.bias.data, bit))
    try:
        yield
    finally:
        with torch.no_grad():
            for module, params in originals.items():
                module.weight.data.copy_(params["weight"])
                if module.bias is not None and params["bias"] is not None:
                    module.bias.data.copy_(params["bias"])


def apply_mixed_precision_allocation(module_to_bit: Dict[nn.Module, int]) -> None:
    with torch.no_grad():
        for module, bit in module_to_bit.items():
            module.weight.data.copy_(quantize_tensor(module.weight.data, bit))
            if module.bias is not None:
                module.bias.data.copy_(quantize_tensor(module.bias.data, bit))
