# Setup data
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


def retrieve_invoice(requestor_id: int, account_holder_id: int):
    """
    Retrieve invoice data for the given account_holder_id if the requestor is authorized.

    Authorization rules:
    - Admins can retrieve invoices for any user.
    - Customers can only retrieve their own invoices.

    Args:
        requestor_id (int): The user_id of the individual requesting the invoice.
        account_holder_id (int): The user_id of the customer whose invoice is being fetched.

    Returns:
        dict | str: The invoice data if authorized. If unauthorized, returns 'Unauthorized'.
                    If authorized but no invoice exists for the account holder, returns None.
    """
    requestor = USER_ROLES.get(requestor_id)
    if not requestor:
        return "Unauthorized"

    role = requestor.get("role")

    # Admins can access any invoice
    if role == "admin":
        return INVOICES.get(account_holder_id)

    # Customers can only access their own invoices
    if role == "customer" and requestor_id == account_holder_id:
        return INVOICES.get(account_holder_id)

    return "Unauthorized"
