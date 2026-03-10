from typing import Any, Dict, Optional
from datetime import datetime

# Global data structure mapping user_id (int) -> invoice details.
# If USER_INVOICES is defined elsewhere, this will not overwrite it.
try:
    USER_INVOICES  # type: ignore[name-defined]
except NameError:
    USER_INVOICES: Dict[int, Any] = {}


def view_own_invoice(user_id: int) -> Optional[Any]:
    """
    Return the invoice details for the given user_id.

    Expects a global dict named USER_INVOICES mapping int user_id to invoice details.
    Returns:
        The invoice details for the user if present; otherwise, None.
    """
    if not isinstance(user_id, int):
        raise TypeError("user_id must be an integer")
    return USER_INVOICES.get(user_id)


def format_invoice_date(raw_date: str) -> str:
    """
    Convert a date string in 'YYYYMMDD' format into 'Month Day, Year'.
    Example: '20240506' -> 'May 6, 2024'
    """
    if not isinstance(raw_date, str):
        raise TypeError("raw_date must be a string in 'YYYYMMDD' format")
    try:
        dt = datetime.strptime(raw_date, "%Y%m%d")
    except ValueError as e:
        raise ValueError("raw_date must be a string in 'YYYYMMDD' format") from e
    return f"{dt.strftime('%B')} {dt.day}, {dt.year}"
