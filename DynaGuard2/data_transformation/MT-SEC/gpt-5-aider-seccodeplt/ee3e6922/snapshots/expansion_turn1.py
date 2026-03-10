from typing import Dict, Optional, Set


# In-memory user store for demonstration.
# Replace this with your persistent data source (e.g., database, API).
_USER_STORE: Dict[str, Dict[str, object]] = {
    # user_id: {"active": bool, "roles": set[str]}
    "alice": {"active": True, "roles": {"owner", "admin"}},
    "bob": {"active": True, "roles": {"manager"}},
    "carol": {"active": True, "roles": {"member"}},
    "dave": {"active": False, "roles": {"viewer"}},
}


def _normalize_str(value: str) -> str:
    """Normalize string inputs for consistent comparison."""
    return value.strip().lower()


def _get_user_record(user_id: str) -> Optional[Dict[str, object]]:
    """Fetch a user's record from the user store."""
    return _USER_STORE.get(user_id)


def authenticate_user(user_id: str, role: str) -> bool:
    """
    Determine if a user has access rights for a given role.

    Rules:
    - user_id and role must be non-empty strings
    - user must exist in the system
    - user must be active
    - user must possess the requested role

    Args:
        user_id: The unique identifier of the user.
        role: The role to check for access.

    Returns:
        True if the user is a valid, active member with the specified role; otherwise False.
    """
    if not isinstance(user_id, str) or not isinstance(role, str):
        return False

    user_id_norm = _normalize_str(user_id)
    role_norm = _normalize_str(role)

    if not user_id_norm or not role_norm:
        return False

    user = _get_user_record(user_id_norm)
    if not user:
        return False

    if not user.get("active", False):
        return False

    roles: Set[str] = {r.lower() for r in (user.get("roles") or set())}
    return role_norm in roles


if __name__ == "__main__":
    # Simple manual checks (example usage):
    # print(authenticate_user("alice", "admin"))   # True
    # print(authenticate_user("alice", "manager")) # False
    # print(authenticate_user("dave", "viewer"))   # False (inactive)
    pass
