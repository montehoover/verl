from typing import Dict, TypedDict


class User(TypedDict):
    user_id: int
    role: str


USER_ROLES: Dict[int, User] = {
    1: {"user_id": 1, "role": "admin"},
    2: {"user_id": 2, "role": "customer"},
    3: {"user_id": 3, "role": "customer"},
    4: {"user_id": 4, "role": "customer"},
}


class Invoice(TypedDict):
    invoice_id: int
    amount: str
    details: str


INVOICES: Dict[int, Invoice] = {
    2: {"invoice_id": 1001, "amount": "$200", "details": "Invoice 1001 for User 2"},
    3: {"invoice_id": 1002, "amount": "$150", "details": "Invoice 1002 for User 3"},
    4: {"invoice_id": 1003, "amount": "$300", "details": "Invoice 1003 for User 4"},
}

_ALLOWED_ROLES = {"admin", "customer"}


def is_admin_or_customer(user_id: int) -> bool:
    if not isinstance(user_id, int):
        return False

    user = USER_ROLES.get(user_id)
    if user is None:
        return False

    role = user.get("role")
    if not isinstance(role, str):
        return False

    return role.lower() in _ALLOWED_ROLES


def can_access_invoice(user_id: int, customer_id: int) -> bool:
    if not isinstance(user_id, int) or not isinstance(customer_id, int):
        return False

    if not is_admin_or_customer(user_id):
        return False

    requester = USER_ROLES.get(user_id)
    if requester is None:
        return False

    role = requester.get("role")
    if not isinstance(role, str):
        return False
    role = role.lower()

    if role == "admin":
        return True

    # For customers, allow access only to their own invoices and only if the target is a valid customer.
    target = USER_ROLES.get(customer_id)
    if target is None:
        return False

    target_role = target.get("role")
    if not isinstance(target_role, str):
        return False

    if target_role.lower() != "customer":
        return False

    return user_id == customer_id


def fetch_invoice(user_id: int, customer: int):
    if not can_access_invoice(user_id, customer):
        return "Unauthorized"
    return INVOICES.get(customer)
