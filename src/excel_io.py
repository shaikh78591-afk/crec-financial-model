from typing import Any, Dict
import pandas as pd

from crec_report_fixed import load_excel as _load_excel  # type: ignore


def load_workbook() -> Any:
    return _load_excel()


def read_sheet(xl, name: str, **kwargs) -> pd.DataFrame:
    return pd.read_excel(xl, name, **kwargs)


