from typing import Dict, Any

# Sample user data for demonstration purposes
USER_DATA: Dict[str, Dict[str, Any]] = {
    "u123": {"name": "Alice", "age": 30, "email": "alice@example.com"},
    "u456": {"name": "Bob", "age": 25, "email": "bob@example.com"},
    "u789": {"name": "Charlie", "age": 28, "email": "charlie@example.com"},
}

def get_user_profile(user_id: str) -> Dict[str, Any]:
    """
    Retrieve the user profile from USER_DATA by user_id.

    Args:
        user_id: The ID of the user whose profile to retrieve.

    Returns:
        The profile data as a dictionary.

    Raises:
        KeyError: If the user_id does not exist in USER_DATA.
    """
    return USER_DATA[user_id]
