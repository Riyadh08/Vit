from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from models import ComponentSpec
from quantization import temporary_quantized_modules

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    def tqdm(iterable, **kwargs):
        return iterable


def kl_temp(logits_ref: torch.Tensor, logits_q: torch.Tensor, temperature: float) -> torch.Tensor:
    t = max(float(temperature), 1e-8)
    log_p = F.log_softmax(logits_ref / t, dim=-1)
    log_q = F.log_softmax(logits_q / t, dim=-1)
    p = log_p.exp()
    return torch.sum(p * (log_p - log_q), dim=-1)


def cache_full_precision_logits(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> List[torch.Tensor]:
    outs: List[torch.Tensor] = []
    with torch.inference_mode():
        for x in loader:
            x = x.to(device)
            z = model(x)
            outs.append(z.detach().cpu())
    return outs


def eval_component_bit_omega(
    model: torch.nn.Module,
    comp: ComponentSpec,
    bit: int,
    loader: DataLoader,
    cached_logits: List[torch.Tensor],
    device: torch.device,
    temperature: float,
) -> float:
    total_kl = 0.0
    total_n = 0
    with torch.inference_mode(), temporary_quantized_modules({comp.module: bit}):
        for batch_idx, x in enumerate(loader):
            x = x.to(device)
            z_q = model(x)
            z_ref = cached_logits[batch_idx].to(device)
            kl = kl_temp(z_ref, z_q, temperature)
            total_kl += float(torch.sum(kl).item())
            total_n += int(kl.numel())
    return total_kl / max(total_n, 1)


def compute_omega_subset_with_cached_baseline(
    model: torch.nn.Module,
    components: List[ComponentSpec],
    bits: List[int],
    loader: DataLoader,
    cached_logits: List[torch.Tensor],
    device: torch.device,
    temperature: float,
    omega_map: Dict[str, Dict[int, float]],
) -> Dict[str, Dict[int, float]]:
    for comp in tqdm(components, desc="Refine components", leave=False):
        omega_map.setdefault(comp.name, {})
        for b in tqdm(bits, desc=f"{comp.name} bits", leave=False):
            omega_map[comp.name][b] = eval_component_bit_omega(
                model, comp, b, loader, cached_logits, device, temperature
            )
    return omega_map


def compute_omega_map(
    model: torch.nn.Module,
    components: List[ComponentSpec],
    bits: List[int],
    loader: DataLoader,
    device: torch.device,
    temperature: float,
) -> Dict[str, Dict[int, float]]:
    cached = cache_full_precision_logits(model, loader, device)
    omega_map: Dict[str, Dict[int, float]] = {}
    for comp in tqdm(components, desc="Components", leave=False):
        omega_map[comp.name] = {}
        for b in tqdm(bits, desc=f"{comp.name} bits", leave=False):
            omega_map[comp.name][b] = eval_component_bit_omega(
                model, comp, b, loader, cached, device, temperature
            )
    return omega_map


def omega_at_assigned_bits(
    omega_map: Dict[str, Dict[int, float]],
    allocation: Dict[str, int],
) -> Dict[str, float]:
    return {name: omega_map[name][allocation[name]] for name in allocation}


def topk_by_fragility(current_omega: Dict[str, float], k: int) -> List[str]:
    ranked = sorted(current_omega.items(), key=lambda x: x[1], reverse=True)
    return [name for name, _ in ranked[:k]]
