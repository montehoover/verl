from typing import Dict, Optional, Set

# Local in-memory user dictionary
_USERS: Dict[str, Dict[str, str]] = {
    "user_1": {"name": "Alice Johnson", "email": "alice@example.com"},
    "user_2": {"name": "Bob Smith", "email": "bob@example.com"},
    "user_3": {"name": "Charlie Parker", "email": "charlie@example.com"},
}

# Predefined list of trusted user IDs
_TRUSTED_USER_IDS: Set[str] = {"user_1", "user_2"}


def get_user_profile(user_id: str) -> Optional[Dict[str, str]]:
    """
    Retrieve a user profile from the local dictionary only if the user_id is trusted.

    Args:
        user_id: The user's unique identifier.

    Returns:
        A dictionary with the user's name and email if the user_id is trusted
        and found; otherwise None.
    """
    if user_id not in _TRUSTED_USER_IDS:
        return None
    return _USERS.get(user_id)
