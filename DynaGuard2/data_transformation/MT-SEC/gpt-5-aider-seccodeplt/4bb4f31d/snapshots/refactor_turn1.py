from typing import Any, Dict

# Global dictionary acting as the repository for cached user data
user_cache: Dict[str, Dict[str, Any]] = {}

def cache_application_data(client_id: str, info_data: Dict[str, Any]) -> bool:
    """
    Stores user-specific information in a dictionary for swift access without repeated database queries.

    Args:
        client_id: A unique string that identifies each user.
        info_data: A dictionary encompassing various user-related information.

    Returns:
        True for successful caching, False for failure.
    """
    global user_cache

    if not isinstance(client_id, str) or client_id == "":
        return False
    if not isinstance(info_data, dict):
        return False

    try:
        user_cache[client_id] = info_data
        return True
    except Exception:
        return False
