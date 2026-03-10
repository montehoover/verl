from typing import Optional, Dict

# Local in-memory user store
_USERS: Dict[str, Dict[str, str]] = {
    "u1": {"name": "Alice Smith", "email": "alice@example.com"},
    "u2": {"name": "Bob Johnson", "email": "bob@example.com"},
    "u3": {"name": "Charlie Davis", "email": "charlie@example.com"},
}


def get_user_profile(uid: str) -> Optional[Dict[str, str]]:
    """
    Retrieve a user profile by user ID.

    Args:
        uid: The user ID as a string.

    Returns:
        A dict with keys 'name' and 'email' if found, otherwise None.
    """
    profile = _USERS.get(uid)
    if profile is None:
        return None
    return {"name": profile["name"], "email": profile["email"]}
