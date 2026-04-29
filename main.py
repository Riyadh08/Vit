from typing import Dict

from config import ExperimentConfig
from dp_solver import budget_from_config, build_cost_value_tables, solve_mckp
from fragility import (
    cache_full_precision_logits,
    compute_omega_map,
    compute_omega_subset_with_cached_baseline,
    omega_at_assigned_bits,
    topk_by_fragility,
)
from models import extract_vit_components, load_vit_model
from quantization import apply_mixed_precision_allocation
from utils import (
    build_calibration_loader,
    get_device,
    print_allocation,
    print_omega_table,
    save_json,
    set_seed,
)


def _map_name_to_module(components):
    return {c.name: c.module for c in components}


def _map_name_to_params(components):
    return {c.name: c.num_params for c in components}


def _setup(cfg: ExperimentConfig):
    set_seed(cfg.synthetic_seed)
    device = get_device()
    model = load_vit_model(cfg.model_name, device)
    components = extract_vit_components(model)
    names = [c.name for c in components]
    loader = build_calibration_loader(
        cfg.calib_image_folder,
        cfg.max_calib_samples,
        cfg.batch_size,
        cfg.synthetic_shape,
        cfg.synthetic_seed,
        cfg.imagenet_mean,
        cfg.imagenet_std,
    )
    return device, model, components, names, loader


def _first_pass(cfg: ExperimentConfig, model, components, names, loader, device):
    name_to_params = _map_name_to_params(components)
    omega_1 = compute_omega_map(
        model, components, cfg.bit_choices, loader, device, cfg.temperature
    )
    if cfg.print_per_component_omega:
        print_omega_table(omega_1)
    costs, values = build_cost_value_tables(
        names, name_to_params, omega_1, cfg.bit_choices, cfg.dp_scale_factor
    )
    budget = budget_from_config(
        names, name_to_params, cfg.avg_bit_target, cfg.dp_scale_factor, cfg.fixed_budget
    )
    alloc_1, total_1, score_1 = solve_mckp(names, cfg.bit_choices, costs, values, budget)
    print_allocation("Allocation before refinement", alloc_1, total_1, budget)
    return alloc_1, total_1, score_1, budget, omega_1


def _refine_pass(
    cfg: ExperimentConfig,
    model,
    components,
    names,
    loader,
    device,
    alloc_1,
    omega_1,
):
    name_to_params = _map_name_to_params(components)
    name_to_module = _map_name_to_module(components)
    current = omega_at_assigned_bits(omega_1, alloc_1)
    print("\nAssigned-bit Omega after first pass:")
    for name in sorted(current):
        print(f"  {name}: {current[name]:.6f}")
    topk = topk_by_fragility(current, cfg.topk_refine)
    apply_mixed_precision_allocation({name_to_module[n]: alloc_1[n] for n in names})
    cached_partial = cache_full_precision_logits(model, loader, device)
    topk_set = set(topk)
    topk_components = [c for c in components if c.name in topk_set]
    omega_2 = {n: dict(omega_1[n]) for n in names}
    omega_2 = compute_omega_subset_with_cached_baseline(
        model,
        topk_components,
        cfg.bit_choices,
        loader,
        cached_partial,
        device,
        cfg.temperature,
        omega_2,
    )
    fixed = {n: alloc_1[n] for n in names if n not in topk}
    costs2, values2 = build_cost_value_tables(
        names, name_to_params, omega_2, cfg.bit_choices, cfg.dp_scale_factor
    )
    return omega_2, topk, fixed, costs2, values2


def run_pipeline(cfg: ExperimentConfig) -> None:
    device, model, components, names, loader = _setup(cfg)
    alloc_1, total_1, score_1, budget, omega_1 = _first_pass(
        cfg, model, components, names, loader, device
    )
    omega_2, topk, fixed, costs2, values2 = _refine_pass(
        cfg, model, components, names, loader, device, alloc_1, omega_1
    )
    alloc_2, total_2, score_2 = solve_mckp(
        names, cfg.bit_choices, costs2, values2, budget, fixed
    )
    print_allocation("Allocation after refinement", alloc_2, total_2, budget)
    final_costs = {name: costs2[name][alloc_2[name]] for name in names}
    payload: Dict = {
        "model_name": cfg.model_name,
        "temperature": cfg.temperature,
        "dp_scale_factor": cfg.dp_scale_factor,
        "budget": budget,
        "topk_components": topk,
        "allocation_before_refinement": alloc_1,
        "allocation_after_refinement": alloc_2,
        "score_before": score_1,
        "score_after": score_2,
        "total_cost_before": total_1,
        "total_cost_after": total_2,
        "per_component_scaled_cost": final_costs,
        "omega_at_assigned_after_first_pass": omega_at_assigned_bits(omega_2, alloc_1),
    }
    save_json(cfg.save_allocation_json_path, payload)
    print(f"Saved allocation JSON to {cfg.save_allocation_json_path}")


if __name__ == "__main__":
    run_pipeline(ExperimentConfig())
