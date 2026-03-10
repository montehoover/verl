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
