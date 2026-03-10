from typing import List, Dict, Any

# Global list to store user information
USER_STORE: List[Dict[str, Any]] = []

# Global dictionary to cache user profile data for quick access
user_cache: Dict[str, Any] = {}


def store_user_info(user_id: str, user_name: str, user_details: Dict[str, Any]) -> bool:
    """
    Store user information in the global USER_STORE list.

    Args:
        user_id (str): The user's ID.
        user_name (str): The user's name.
        user_details (Dict[str, Any]): Additional user details, e.g., {"email": "...", "age": 30}

    Returns:
        bool: True if the information was stored successfully, False otherwise.
    """
    if not isinstance(user_id, str) or not isinstance(user_name, str) or not isinstance(user_details, dict):
        return False

    try:
        USER_STORE.append({
            "user_id": user_id,
            "user_name": user_name,
            "user_details": dict(user_details)
        })
        return True
    except Exception:
        return False


def cache_profile_data(profile_id: str, profile_data: Dict[str, Any]) -> bool:
    """
    Cache user profile data in the global user_cache dictionary.

    Args:
        profile_id (str): Unique identifier for the user profile.
        profile_data (Dict[str, Any]): User profile details to cache.

    Returns:
        bool: True if the profile data was cached successfully, False otherwise.
    """
    if not isinstance(profile_id, str) or not isinstance(profile_data, dict):
        return False

    try:
        user_cache[profile_id] = dict(profile_data)
        return True
    except Exception:
        return False
