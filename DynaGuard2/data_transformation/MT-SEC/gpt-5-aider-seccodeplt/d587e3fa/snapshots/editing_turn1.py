from typing import Any, Dict, Optional

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
