from typing import Any, Dict, Tuple

# Thin facade over current monolith to enable imports without breaking
from crec_report_fixed import build_scenario as _build_scenario  # type: ignore
from crec_report_fixed import compute_scenario_metrics as _compute_metrics  # type: ignore
from crec_report_fixed import simulate_mass_balance as _simulate_mass_balance  # type: ignore


def build_scenario(msw_value: float, tires_value: float, price_value: float, inflation_value: float, capex_multiplier: float) -> Dict[str, Any]:
    return _build_scenario(msw_value, tires_value, price_value, inflation_value, capex_multiplier)


def compute_scenario_metrics(data: Dict[str, Any], capex_value: float) -> Dict[str, float]:
    return _compute_metrics(data, capex_value)


def simulate_mass_balance(msw_tpd: float, tires_tpd: float) -> Tuple[float, float, float, float]:
    return _simulate_mass_balance(msw_tpd, tires_tpd)



