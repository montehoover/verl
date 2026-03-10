from typing import Dict, Any

existing_user_ids = []
cache_storage: Dict[str, Dict[str, Any]] = {}
user_cache = cache_storage

def create_user_profile(uid: str, details: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create a user profile dictionary that encapsulates a unique identifier and user attributes.
    """
    if not isinstance(uid, str) or not uid:
        raise ValueError("uid must be a non-empty string")
    if not isinstance(details, dict):
        raise TypeError("details must be a dict")

    return {
        "uid": uid,
        "details": dict(details),
    }

def assign_user_id(profile: Dict[str, Any]) -> Dict[str, Any]:
    """
    Attach a unique user_id to the provided user profile and track it globally.
    If the profile already has a user_id, it will be recorded if not already tracked.
    """
    if not isinstance(profile, dict):
        raise TypeError("profile must be a dict")

    global existing_user_ids

    # If user_id already present, ensure it's tracked and return
    current_id = profile.get("user_id")
    if isinstance(current_id, str) and current_id:
        if current_id not in existing_user_ids:
            existing_user_ids.append(current_id)
        return profile

    # Generate a new unique user_id
    idx = len(existing_user_ids) + 1
    new_id = f"user_{idx}"
    while new_id in existing_user_ids:
        idx += 1
        new_id = f"user_{idx}"

    profile["user_id"] = new_id
    existing_user_ids.append(new_id)
    return profile

def store_user_data(uid: str, data: Dict[str, Any]) -> bool:
    """
    Store user-specific data in the in-memory cache for quick access.
    Returns True if caching succeeded.
    """
    if not isinstance(uid, str) or not uid:
        raise ValueError("uid must be a non-empty string")
    if not isinstance(data, dict):
        raise TypeError("data must be a dict")

    global cache_storage
    try:
        cache_storage[uid] = dict(data)
        return True
    except Exception:
        return False
