from typing import Dict, Any

# Global in-memory cache for user profiles
user_cache: Dict[str, Dict[str, Any]] = {}

def cache_profile_data(profile_id: str, profile_data: dict) -> bool:
    """
    Cache user-specific profile data in memory.

    Args:
        profile_id (str): A unique string identifier for each user.
        profile_data (dict): A dictionary containing various details about the user.

    Returns:
        bool: True if the caching operation succeeds, False if it fails.
    """
    global user_cache

    # Validate input types
    if not isinstance(profile_id, str) or not isinstance(profile_data, dict):
        return False

    try:
        # Store a shallow copy to prevent accidental external mutation of the cached data
        user_cache[profile_id] = dict(profile_data)
        return True
    except Exception:
        return False
