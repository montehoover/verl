USER_ROLES = {
    1: {"user_id": 1, "role": "admin"},
    2: {"user_id": 2, "role": "customer"},
    3: {"user_id": 3, "role": "customer"},
    4: {"user_id": 4, "role": "customer"}
}

INVOICES = {
    2: {"invoice_id": 1001, "amount": "$200", "details": "Invoice 1001 for User 2"},
    3: {"invoice_id": 1002, "amount": "$150", "details": "Invoice 1002 for User 3"},
    4: {"invoice_id": 1003, "amount": "$300", "details": "Invoice 1003 for User 4"}
}

def fetch_invoice(user_id: int, customer: int):
    """
    Fetch the invoice for the specified customer if the requesting user is authorized.
    - Admin users can fetch any customer's invoice.
    - Customers can fetch only their own invoice.
    Returns the invoice dict if accessible, 'Invoice not found' if authorized but missing, otherwise 'Unauthorized'.
    """
    user = USER_ROLES.get(user_id)
    if not user:
        return 'Unauthorized'

    is_admin = user.get('role') == 'admin'
    is_self = user_id == customer

    if not (is_admin or is_self):
        return 'Unauthorized'

    invoice = INVOICES.get(customer)
    if invoice is None:
        return 'Invoice not found'
    return invoice
