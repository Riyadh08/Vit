from typing import Dict, List, Optional, Tuple


def build_cost_value_tables(
    component_names: List[str],
    num_params: Dict[str, int],
    omega_map: Dict[str, Dict[int, float]],
    bits: List[int],
    scale_factor: int,
) -> Tuple[Dict[str, Dict[int, int]], Dict[str, Dict[int, float]]]:
    costs, values = {}, {}
    for name in component_names:
        costs[name], values[name] = {}, {}
        base = omega_map[name][2]
        for b in bits:
            scaled = (num_params[name] // scale_factor) * b
            costs[name][b] = max(1, int(scaled))
            values[name][b] = float(base - omega_map[name][b])
    return costs, values


def solve_mckp(
    component_names: List[str],
    bits: List[int],
    costs: Dict[str, Dict[int, int]],
    values: Dict[str, Dict[int, float]],
    budget: int,
    fixed_alloc: Optional[Dict[str, int]] = None,
) -> Tuple[Dict[str, int], int, float]:
    fixed_alloc = fixed_alloc or {}
    n = len(component_names)
    neg = -1e18
    dp = [[neg] * (budget + 1) for _ in range(n + 1)]
    parent = [[None] * (budget + 1) for _ in range(n + 1)]
    dp[0][0] = 0.0
    for i, name in enumerate(component_names, start=1):
        allowed = [fixed_alloc[name]] if name in fixed_alloc else bits
        for w in range(budget + 1):
            best, best_prev = neg, None
            for b in allowed:
                c = costs[name][b]
                if c <= w and dp[i - 1][w - c] > neg:
                    cand = dp[i - 1][w - c] + values[name][b]
                    if cand > best:
                        best, best_prev = cand, (w - c, b)
            dp[i][w] = best
            parent[i][w] = best_prev
    best_w = max(range(budget + 1), key=lambda x: dp[n][x])
    alloc = {}
    w = best_w
    for i in range(n, 0, -1):
        prev = parent[i][w]
        if prev is None:
            raise RuntimeError("No feasible MCKP allocation found")
        pw, b = prev
        alloc[component_names[i - 1]] = b
        w = pw
    total = sum(costs[name][alloc[name]] for name in component_names)
    return alloc, total, float(dp[n][best_w])


def budget_from_config(
    component_names: List[str],
    num_params: Dict[str, int],
    avg_bit_target: int,
    scale_factor: int,
    fixed_budget: Optional[int],
) -> int:
    if fixed_budget is not None:
        return int(fixed_budget)
    total_params = sum(num_params[name] for name in component_names)
    return int((total_params // scale_factor) * avg_bit_target)
