from typing import Dict, Set


# In-memory user directory mapping user_id -> set of roles.
# Roles are stored in lowercase for normalization.
USERS: Dict[str, Set[str]] = {
    "alice": {"customer"},
    "bob": {"customer", "teller"},
    "carol": {"manager"},
    "admin": {"admin"},
}


def authenticate_user(user_id: str, role: str) -> bool:
    """
    Authenticate a user by verifying their user_id exists and has the specified role.

    Args:
        user_id: The unique identifier for the user (exact match, trimmed; case-sensitive).
        role: The role to validate against (trimmed; case-insensitive).

    Returns:
        True if the user exists and has the specified role, otherwise False.
    """
    if not isinstance(user_id, str) or not isinstance(role, str):
        return False

    normalized_user_id = user_id.strip()
    normalized_role = role.strip().lower()

    if not normalized_user_id or not normalized_role:
        return False

    assigned_roles = USERS.get(normalized_user_id)
    if not assigned_roles:
        return False

    return normalized_role in assigned_roles
