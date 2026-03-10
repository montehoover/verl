from typing import Dict

# Global dictionary to store user data mapping user_id -> user_info dict
USER_DATA: Dict[str, Dict[str, str]] = {}

# Global cache for application data mapping client_id -> info_data dict
user_cache: Dict[str, dict] = {}


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


def cache_application_data(client_id: str, info_data: dict) -> bool:
    """
    Cache application/user data in the global user_cache dictionary.

    Args:
        client_id: The unique user/client identifier as a string.
        info_data: A dictionary with user-related information to cache.

    Returns:
        True if data is cached successfully, False otherwise.
    """
    try:
        if not isinstance(client_id, str) or not isinstance(info_data, dict):
            return False
        user_cache[client_id] = info_data
        return True
    except Exception:
        return False
