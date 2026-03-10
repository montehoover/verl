from typing import Dict, Any

# Global cache for user data
user_cache: Dict[str, Dict[str, Any]] = {}


def create_user_profile(client_id: str, info_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create a local user profile by combining the client_id with the provided info_data.

    The returned dictionary includes:
    - all key/value pairs from info_data
    - a "client_id" field set to the provided client_id (overrides any existing key)
    """
    if not isinstance(client_id, str):
        raise TypeError("client_id must be a string")
    if not isinstance(info_data, dict):
        raise TypeError("info_data must be a dictionary")

    # Merge info_data with client_id, ensuring client_id takes precedence
    profile: Dict[str, Any] = {**info_data, "client_id": client_id}
    return profile


def update_user_info(profile: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    """
    Update a user profile with new information.

    - Returns a new dictionary; does not mutate the inputs.
    - The "client_id" field is preserved and cannot be changed via updates.
    """
    if not isinstance(profile, dict):
        raise TypeError("profile must be a dictionary")
    if not isinstance(updates, dict):
        raise TypeError("updates must be a dictionary")

    updated_profile: Dict[str, Any] = dict(profile)

    # Prevent changing the client_id through updates
    sanitized_updates = {k: v for k, v in updates.items() if k != "client_id"}

    updated_profile.update(sanitized_updates)
    return updated_profile


def cache_application_data(client_id: str, info_data: Dict[str, Any]) -> bool:
    """
    Cache user-specific information in the global user_cache for quick access.

    Stores a full user profile (including client_id) under the client_id key.

    Returns:
        True if the data was cached successfully.

    Raises:
        TypeError: If input types are invalid.
    """
    if not isinstance(client_id, str):
        raise TypeError("client_id must be a string")
    if not isinstance(info_data, dict):
        raise TypeError("info_data must be a dictionary")

    profile = create_user_profile(client_id, info_data)
    # Store a shallow copy to avoid external mutations affecting the cache
    user_cache[client_id] = dict(profile)
    return True
