from typing import Any, Dict

from crec_report_fixed import optimize_scenario as _optimize_scenario  # type: ignore
from crec_report_fixed import generate_optimizer_explanation as _gen_expl  # type: ignore


def optimize_scenario(target_irr: float, sample_count: int, tolerance: float, seed: int | None = None) -> Dict[str, Any]:
    return _optimize_scenario(target_irr, sample_count, tolerance, seed=seed)


def generate_optimizer_explanation(result: Dict[str, Any]) -> str:
    return _gen_expl(result)


