from typing import Any, Dict, Optional
from datetime import datetime

# Global data structure mapping user_id (int) -> invoice details.
# If USER_INVOICES is defined elsewhere, this will not overwrite it.
try:
    USER_INVOICES  # type: ignore[name-defined]
except NameError:
    USER_INVOICES: Dict[int, Any] = {}

# Global data structures for roles and invoices.
# If these are defined elsewhere, this will not overwrite them.
try:
    USER_ROLES  # type: ignore[name-defined]
except NameError:
    USER_ROLES: Dict[int, Dict[str, Any]] = {}

try:
    INVOICES  # type: ignore[name-defined]
except NameError:
    INVOICES: Dict[int, Dict[str, Any]] = {}


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


def access_invoice(user_attempt_id: int, cust_id: int) -> Any:
    """
    Retrieve an invoice if the requesting user is authorized.

    Rules:
      - Admins can access any customer's invoice.
      - Customers can only access their own invoice.
    Returns:
      - The invoice data (from INVOICES) if authorized.
      - The string 'Unauthorized' if not authorized.
    """
    if not isinstance(user_attempt_id, int):
        raise TypeError("user_attempt_id must be an integer")
    if not isinstance(cust_id, int):
        raise TypeError("cust_id must be an integer")

    role_info = USER_ROLES.get(user_attempt_id)
    if not isinstance(role_info, dict) or "role" not in role_info:
        return "Unauthorized"

    role = role_info.get("role")
    is_admin = role == "admin"
    is_self = role == "customer" and user_attempt_id == cust_id

    if is_admin or is_self:
        return INVOICES.get(cust_id)

    return "Unauthorized"
