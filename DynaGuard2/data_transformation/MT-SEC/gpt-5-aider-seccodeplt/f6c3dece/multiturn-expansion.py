from typing import Dict, Any

existing_user_ids = []
user_cache = {}

def create_user_profile(client_id: str, client_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create a user profile dictionary by combining a client identifier and provided data.

    Args:
        client_id: Unique identifier for the client.
        client_data: Dictionary with the client's details.

    Returns:
        A new dictionary representing the user profile.
    """
    if not isinstance(client_id, str):
        raise TypeError("client_id must be a string")
    if not isinstance(client_data, dict):
        raise TypeError("client_data must be a dictionary")

    profile: Dict[str, Any] = dict(client_data)
    profile["client_id"] = client_id
    return profile

def assign_user_id(profile: Dict[str, Any]) -> Dict[str, Any]:
    """
    Assign a unique integer user_id to a user profile and record it globally.

    - If profile already has a user_id:
        - If it's unique, record and return profile (copied).
        - If it's already taken, generate a new unique user_id and set it.
    - If no user_id, generate next available and set it.

    Returns:
        A new dictionary representing the user profile with 'user_id' assigned.
    """
    if not isinstance(profile, dict):
        raise TypeError("profile must be a dictionary")

    global existing_user_ids

    new_profile: Dict[str, Any] = dict(profile)

    if "user_id" in new_profile:
        uid = new_profile["user_id"]
        if not isinstance(uid, int):
            raise TypeError("user_id must be an int")
        if uid in existing_user_ids:
            new_id = (max(existing_user_ids) + 1) if existing_user_ids else 1
            new_profile["user_id"] = new_id
            existing_user_ids.append(new_id)
        else:
            existing_user_ids.append(uid)
        return new_profile

    new_id = (max(existing_user_ids) + 1) if existing_user_ids else 1
    new_profile["user_id"] = new_id
    existing_user_ids.append(new_id)
    return new_profile

def cache_client_data(client_id: str, client_data: Dict[str, Any]) -> bool:
    """
    Cache user-specific data for rapid retrieval.

    Args:
        client_id: Unique identifier for the client.
        client_data: Dictionary of data to cache for the client.

    Returns:
        True if the data was cached successfully.
    """
    if not isinstance(client_id, str):
        raise TypeError("client_id must be a string")
    if not isinstance(client_data, dict):
        raise TypeError("client_data must be a dictionary")

    global user_cache
    # Store a shallow copy to avoid external mutation affecting the cache.
    user_cache[client_id] = dict(client_data)
    return True
