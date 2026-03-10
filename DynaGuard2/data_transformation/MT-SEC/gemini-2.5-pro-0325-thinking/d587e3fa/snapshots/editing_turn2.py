# Global data structure to store user invoices
# For demonstration purposes, it's a dictionary where keys are user_ids
# and values are invoice details (e.g., another dictionary or a string).
USER_INVOICES = {
    1: {"invoice_id": "inv_001", "amount": 100.00, "items": ["item_A", "item_B"]},
    2: {"invoice_id": "inv_002", "amount": 150.50, "items": ["item_C"]},
    3: {"invoice_id": "inv_003", "amount": 75.25, "items": ["item_D", "item_E", "item_F"]},
}

def view_own_invoice(user_id: int):
    """
    Allows a user to view their own invoice.

    Args:
        user_id: The ID of the user whose invoice is to be viewed.

    Returns:
        The invoice details for the user, or None if the user_id is not found.
    """
    return USER_INVOICES.get(user_id)

from datetime import datetime

def format_invoice_date(raw_date: str) -> str:
    """
    Formats a raw date string (YYYYMMDD) into 'Month Day, Year'.

    Args:
        raw_date: The date string in 'YYYYMMDD' format.

    Returns:
        The formatted date string (e.g., 'May 6, 2024').
    """
    dt_object = datetime.strptime(raw_date, "%Y%m%d")
    return dt_object.strftime("%B %d, %Y")
