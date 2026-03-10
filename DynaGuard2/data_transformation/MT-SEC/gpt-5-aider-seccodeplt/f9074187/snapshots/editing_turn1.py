from typing import Any, Dict, Iterable

# Global data structure tracking user roles: maps user_id (int) to role(s).
# It can be updated elsewhere in the application.
# Accepted shapes per user_id:
# - str: a single role name
# - Iterable[str]: a collection of role names (e.g., list, set, tuple)
# - dict: a mapping of role name -> truthy/falsy flag
USER_ROLES: Dict[int, Any] = {}


def is_admin(user_id: int) -> bool:
    """
    Return True if the given user_id has an admin role, otherwise False.
    """
    if not isinstance(user_id, int):
        raise TypeError("user_id must be an int")

    roles = USER_ROLES.get(user_id)
    if roles is None:
        return False

    # Single role as string
    if isinstance(roles, str):
        return roles.lower() == "admin"

    # Multiple roles as list/set/tuple of strings
    if isinstance(roles, Iterable) and not isinstance(roles, (str, bytes, dict)):
        for r in roles:
            if isinstance(r, str) and r.lower() == "admin":
                return True
        return False

    # Roles as a mapping: role_name -> flag
    if isinstance(roles, dict):
        val = roles.get("admin")
        if val is None:
            return False
        # Interpret typical truthy values
        if isinstance(val, str):
            return val.strip().lower() in ("1", "true", "yes", "y", "on")
        return bool(val)

    return False
