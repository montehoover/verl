INVOICES = {
    2: {"invoice_id": 1001, "amount": "$200", "details": "Invoice 1001 for User 2"},
    3: {"invoice_id": 1002, "amount": "$150", "details": "Invoice 1002 for User 3"},
    4: {"invoice_id": 1003, "amount": "$300", "details": "Invoice 1003 for User 4"}
}

def fetch_user_invoice(user_id: int):
    """
    Fetch the invoice for the given user_id from the global INVOICES dictionary.
    Returns the invoice dict if found, otherwise the string 'Invoice not found'.
    """
    return INVOICES.get(user_id, 'Invoice not found')
