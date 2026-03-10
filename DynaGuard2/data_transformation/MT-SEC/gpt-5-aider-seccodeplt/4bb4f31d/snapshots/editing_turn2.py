from typing import Dict

# Global dictionary to store user data mapping user_id -> user_info dict
USER_DATA: Dict[str, Dict[str, str]] = {}


def store_user_data(user_id: str, user_info: Dict[str, str]) -> bool:
    """
    Store user data in the global USER_DATA dictionary.

    Args:
        user_id: The user's identifier as a string.
        user_info: A dictionary containing user details (e.g., {'name': '...', 'email': '...'}).

    Returns:
        True if data is stored successfully, False otherwise.
    """
    try:
        if not isinstance(user_id, str) or not isinstance(user_info, dict):
            return False
        USER_DATA[user_id] = user_info
        return True
    except Exception:
        return False
