USER_ROLES = {
    1: {"user_id": 1, "role": "admin"},
    2: {"user_id": 2, "role": "customer"},
    3: {"user_id": 3, "role": "customer"},
    4: {"user_id": 4, "role": "customer"}
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
