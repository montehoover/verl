from typing import Optional, Dict

USERS: Dict[str, Dict[str, str]] = {
    "u123": {"name": "Alice Johnson", "email": "alice.johnson@example.com"},
    "u456": {"name": "Bob Smith", "email": "bob.smith@example.com"},
    "u789": {"name": "Carol Lee", "email": "carol.lee@example.com"},
}


def get_user_profile(user_id: str) -> Optional[Dict[str, str]]:
    """
    Retrieve a user profile from a local dictionary based on a user ID.

    Args:
        user_id: The user's unique identifier.

    Returns:
        A dict with 'name' and 'email' if the user exists, otherwise None.
    """
    return USERS.get(user_id)
