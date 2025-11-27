from typing import Any, Dict

from crec_report_fixed import run_monte_carlo as _run_monte_carlo  # type: ignore


def run_monte_carlo(iterations: int, base_inputs: Dict[str, float], volatilities: Dict[str, float], inflation_value: float, capex_multiplier: float, seed: int | None = None) -> Dict[str, Any]:
    return _run_monte_carlo(iterations, base_inputs, volatilities, inflation_value, capex_multiplier, seed=seed)


