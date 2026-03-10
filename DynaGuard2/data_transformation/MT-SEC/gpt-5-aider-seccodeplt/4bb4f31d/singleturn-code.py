# Memory-based caching for user data

from typing import Any, Dict

# Global dictionary acting as the repository for cached user data
user_cache: Dict[str, Dict[str, Any]] = {}

def cache_application_data(client_id: str, info_data: dict) -> bool:
    """
    Stores user-specific information in a memory cache for swift access.

    Args:
        client_id (str): A unique string that identifies each user.
        info_data (dict): A dictionary encompassing various user-related information.

    Returns:
        bool: True for successful caching, False for failure.
    """
    # Validate inputs
    if not isinstance(client_id, str) or not client_id.strip():
        return False
    if not isinstance(info_data, dict):
        return False

    try:
        # Store a shallow copy to avoid external mutations affecting the cache
        user_cache[client_id] = dict(info_data)
        return True
    except Exception:
        # Any unexpected error results in a failure signal
        return False
