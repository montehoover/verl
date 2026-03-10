from typing import Any, Dict

def store_user_data(uid: str, data: Dict[str, Any]) -> bool:
    """
    Add new user information to the global cache.

    Args:
        uid: Unique string identifier for the user.
        data: Dictionary containing user information to store.

    Returns:
        True if the data was stored successfully; False if constraints prevent storing.
        Constraints:
        - uid must be a non-empty string.
        - data must be a dictionary.
        - uid must not already exist in the cache (no overwriting).
    """
    global cache_storage

    if not isinstance(uid, str) or not uid:
        return False
    if not isinstance(data, dict):
        return False
    if uid in cache_storage:
        return False

    try:
        cache_storage[uid] = dict(data)
        return True
    except Exception:
        return False
