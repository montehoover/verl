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
    Check whether the given user_id corresponds to an admin user.

    Args:
        user_id (int): The ID of the user to check.

    Returns:
        bool: True if the user has the 'admin' role, otherwise False.
    """
    user = USER_ROLES.get(user_id)
    return bool(user and user.get("role") == "admin")


def can_access_invoice(requestor_id: int, account_holder_id: int) -> bool:
    """
    Determine if the requestor can access the account holder's invoice.

    Rules:
    - Admins can access any invoice.
    - Non-admins can access only their own invoice.

    Args:
        requestor_id (int): The ID of the user making the request.
        account_holder_id (int): The ID of the account holder whose invoice is being accessed.

    Returns:
        bool: True if access is allowed, otherwise False.
    """
    if is_admin(requestor_id):
        return True
    return requestor_id == account_holder_id


def retrieve_invoice(requestor_id: int, account_holder_id: int):
    """
    Retrieve the invoice data for the account holder if the requestor is authorized.

    Args:
        requestor_id (int): The ID of the user making the request.
        account_holder_id (int): The ID of the account holder whose invoice is being retrieved.

    Returns:
        dict | str: The invoice data if authorized, otherwise 'Unauthorized'.
    """
    if can_access_invoice(requestor_id, account_holder_id):
        return INVOICES.get(account_holder_id)
    return "Unauthorized"
