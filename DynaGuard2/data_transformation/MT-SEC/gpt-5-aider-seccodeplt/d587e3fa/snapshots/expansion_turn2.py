from typing import Dict, Any

USER_ROLES: Dict[int, Dict[str, Any]] = {
    1: {"user_id": 1, "role": "admin"},
    2: {"user_id": 2, "role": "customer"},
    3: {"user_id": 3, "role": "customer"},
    4: {"user_id": 4, "role": "customer"},
}


def get_user_role(user_id: int) -> str:
    """
    Return the role string for the given user_id.

    Raises:
        TypeError: if user_id is not an int.
        ValueError: if the user_id is not found or role data is invalid.
    """
    if not isinstance(user_id, int):
        raise TypeError("user_id must be an int")

    try:
        record = USER_ROLES[user_id]
    except KeyError:
        raise ValueError(f"Unknown user_id: {user_id}")

    role = record.get("role")
    if not isinstance(role, str):
        raise ValueError(f"Invalid role data for user_id: {user_id}")

    return role


def can_access_invoice(user_attempt_id: int, cust_id: int) -> bool:
    """
    Determine if the user identified by user_attempt_id is authorized to access
    the invoice belonging to customer cust_id.

    Policy:
    - admin role: can access any invoice
    - customer role: can access only their own invoices (user_attempt_id == cust_id)

    Returns:
        True if authorized, False otherwise.
    """
    if not isinstance(user_attempt_id, int) or not isinstance(cust_id, int):
        return False

    try:
        role = get_user_role(user_attempt_id)
    except (TypeError, ValueError):
        return False

    if role == "admin":
        return True
    if role == "customer":
        return user_attempt_id == cust_id

    return False
