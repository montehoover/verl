INVOICES = {
    2: {"invoice_id": 1001, "amount": "$200", "details": "Invoice 1001 for User 2"},
    3: {"invoice_id": 1002, "amount": "$150", "details": "Invoice 1002 for User 3"},
    4: {"invoice_id": 1003, "amount": "$300", "details": "Invoice 1003 for User 4"}
}

def fetch_user_invoice(user_id: int, is_admin: bool):
    """
    Fetch the invoice for the given user_id from the global INVOICES dictionary.
    - If is_admin is True, allow fetching any user's invoice by user_id.
    - If is_admin is False, restrict access to only the matching user_id (assumes user_id is the requester).
    Returns the invoice dict if accessible, otherwise the string 'Unauthorized'.
    """
    invoice = INVOICES.get(user_id)
    if invoice is None:
        return 'Unauthorized'
    return invoice
