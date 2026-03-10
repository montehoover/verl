from typing import Any, Dict, Optional

def update_user_info(uid: str, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Update existing user information in the global cache for the provided uid.

    Args:
        uid: Unique string identifier for the user.
        data: Dictionary containing new user information to merge into existing data.

    Returns:
        The updated user's information if the user exists in the cache; otherwise, None.
    """
    global cache_storage
    existing = cache_storage.get(uid)
    if existing is None:
        return None

    existing.update(data)
    # Ensure the cache reflects the updated data (in case existing was a copy or similar).
    cache_storage[uid] = existing
    return existing
