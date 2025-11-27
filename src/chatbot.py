from typing import Optional

from crec_report_fixed import handle_chat_query as _handle, apply_chat_command as _apply  # type: ignore


def handle_query(query: str) -> None:
    _handle(query)


def apply_command(query: str):
    return _apply(query)


