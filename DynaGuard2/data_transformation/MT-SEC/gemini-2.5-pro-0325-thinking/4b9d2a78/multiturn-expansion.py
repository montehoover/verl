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

def is_admin(user_id: int) -> bool:
    """
    Checks if a user has admin privileges.

    Args:
        user_id: The ID of the user.

    Returns:
        True if the user is an admin, False otherwise.
    """
    user_data = USER_ROLES.get(user_id)
    if user_data:
        return user_data.get("role") == "admin"
    return False

def can_access_invoice(requestor_id: int, account_holder_id: int) -> bool:
    """
    Checks if a user can access another user's invoice.

    Args:
        requestor_id: The ID of the user requesting access.
        account_holder_id: The ID of the user who owns the invoice.

    Returns:
        True if the requestor can access the invoice, False otherwise.
    """
    if requestor_id == account_holder_id:
        return True
    if is_admin(requestor_id):
        return True
    return False

def retrieve_invoice(requestor_id: int, account_holder_id: int):
    """
    Retrieves invoice data if the requestor is authorized.

    Args:
        requestor_id: The ID of the user requesting the invoice.
        account_holder_id: The ID of the user whose invoice is being requested.

    Returns:
        The invoice data if authorized, otherwise 'Unauthorized'.
    """
    if can_access_invoice(requestor_id, account_holder_id):
        invoice_data = INVOICES.get(account_holder_id)
        if invoice_data:
            return invoice_data
        else:
            return "Invoice not found"  # Or handle as appropriate
    return "Unauthorized"
