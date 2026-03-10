from typing import Optional, Dict, List

USERS: Dict[str, Dict[str, str]] = {
    "u123": {"name": "Alice Johnson", "email": "alice.johnson@example.com"},
    "u456": {"name": "Bob Smith", "email": "bob.smith@example.com"},
    "u789": {"name": "Carol Lee", "email": "carol.lee@example.com"},
}


def get_user_profile(user_id: str, trusted_ids: List[str]) -> Optional[Dict[str, str]]:
    """
    Retrieve a user profile from a local dictionary based on a user ID,
    only if the user_id is included in the list of trusted IDs.

    Args:
        user_id: The user's unique identifier.
        trusted_ids: A list of trusted user IDs.

    Returns:
        A dict with 'name' and 'email' if the user exists and is trusted, otherwise None.
    """
    if user_id not in trusted_ids:
        return None
    return USERS.get(user_id)
