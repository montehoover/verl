# Setup data structures mapping user IDs to roles and invoices.
# These are provided as part of the application context.
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
    Retrieve invoice data for a given account holder if authorized.

    Rules:
    - Admins can retrieve invoices for any user.
    - Customers can retrieve only their own invoices.
    - If not authorized, return the string 'Unauthorized'.

    Returns:
        dict | None | str: The invoice data dict if available and authorized,
        None if authorized but no invoice exists for the account holder,
        or 'Unauthorized' if the requester lacks permission.
    """
    requestor = USER_ROLES.get(requestor_id)
    if requestor is None:
        return 'Unauthorized'

    is_admin = requestor.get("role") == "admin"
    is_self = requestor_id == account_holder_id

    if not (is_admin or is_self):
        return 'Unauthorized'

    return INVOICES.get(account_holder_id)
