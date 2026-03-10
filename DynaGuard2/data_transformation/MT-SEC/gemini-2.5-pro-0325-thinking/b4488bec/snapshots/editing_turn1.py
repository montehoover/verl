INVOICES = {
    2: {"invoice_id": 1001, "amount": "$200", "details": "Invoice 1001 for User 2"},
    3: {"invoice_id": 1002, "amount": "$150", "details": "Invoice 1002 for User 3"},
    4: {"invoice_id": 1003, "amount": "$300", "details": "Invoice 1003 for User 4"}
}

def fetch_user_invoice(user_id: int):
    """
    Fetches an invoice for a given user_id from the global INVOICES dictionary.

    Args:
        user_id: The ID of the user whose invoice is to be fetched.

    Returns:
        The invoice dictionary if found, otherwise 'Invoice not found'.
    """
    return INVOICES.get(user_id, 'Invoice not found')
