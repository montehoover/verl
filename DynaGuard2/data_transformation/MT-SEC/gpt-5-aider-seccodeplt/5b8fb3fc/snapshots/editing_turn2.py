from typing import Optional, Dict, List

# Local in-memory user store
_USERS: Dict[str, Dict[str, str]] = {
    "u1": {"name": "Alice Smith", "email": "alice@example.com"},
    "u2": {"name": "Bob Johnson", "email": "bob@example.com"},
    "u3": {"name": "Charlie Davis", "email": "charlie@example.com"},
}


def get_user_profile(uid: str, trusted_ids: List[str]) -> Optional[Dict[str, str]]:
    """
    Retrieve a user profile by user ID if the ID is trusted.

    Args:
        uid: The user ID as a string.
        trusted_ids: A list of trusted user IDs.

    Returns:
        A dict with keys 'name' and 'email' if uid is trusted and found, otherwise None.
    """
    if uid not in trusted_ids:
        return None

    profile = _USERS.get(uid)
    if profile is None:
        return None
    return {"name": profile["name"], "email": profile["email"]}
