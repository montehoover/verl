# In-memory cache for user profiles
user_cache = {}

def cache_profile_data(profile_id: str, profile_data: dict) -> bool:
    """
    Cache user-specific profile data in memory.

    Args:
        profile_id (str): A unique string identifier for each user.
        profile_data (dict): A dictionary containing various details about the user.

    Returns:
        bool: True if the caching operation succeeds, False otherwise.
    """
    try:
        if not isinstance(profile_id, str) or not profile_id:
            return False
        if not isinstance(profile_data, dict):
            return False

        # Store a shallow copy to avoid external mutations affecting the cache
        user_cache[profile_id] = dict(profile_data)
        return True
    except Exception:
        return False
