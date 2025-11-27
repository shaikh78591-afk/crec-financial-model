from typing import Any, Dict, Tuple

from crec_report_fixed import parse_data_sources_switches_and_params as _parse  # type: ignore


def parse_data_sources(df) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    return _parse(df)


