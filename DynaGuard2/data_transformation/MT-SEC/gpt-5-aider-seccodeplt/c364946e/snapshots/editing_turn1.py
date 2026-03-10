from typing import Dict, Optional

# Local in-memory user dictionary
_USERS: Dict[str, Dict[str, str]] = {
    "user_1": {"name": "Alice Johnson", "email": "alice@example.com"},
    "user_2": {"name": "Bob Smith", "email": "bob@example.com"},
    "user_3": {"name": "Charlie Parker", "email": "charlie@example.com"},
}


def get_user_profile(user_id: str) -> Optional[Dict[str, str]]:
    """
    Retrieve a user profile from the local dictionary.

    Args:
        user_id: The user's unique identifier.

    Returns:
        A dictionary with the user's name and email if found; otherwise None.
    """
    return _USERS.get(user_id)
