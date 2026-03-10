from typing import Dict, Any, List
import uuid


existing_profile_ids: List[str] = []
user_cache: Dict[str, Dict[str, Any]] = {}


def create_user_profile(profile_id: str, profile_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create a user profile by combining the given profile_id and profile_data.

    Args:
        profile_id: Unique identifier for the user profile.
        profile_data: Dictionary containing user details.

    Returns:
        A dictionary representing the complete user profile, where the provided
        profile_id is stored under the "id" key. Any "id" present in profile_data
        will be overridden by the profile_id argument.
    """
    profile: Dict[str, Any] = dict(profile_data)
    profile["id"] = profile_id
    return profile


def _generate_unique_profile_id() -> str:
    """
    Generate a unique profile ID that is not already present in existing_profile_ids.
    """
    while True:
        candidate = uuid.uuid4().hex
        if candidate not in existing_profile_ids:
            return candidate


def assign_profile_id(profile: Dict[str, Any]) -> Dict[str, Any]:
    """
    Assign a unique 'profile_id' to the given profile dictionary and record it
    in the global existing_profile_ids list.

    If the profile already contains a 'profile_id' and it's unique, it will be used
    and recorded. Otherwise, a new unique ID will be generated and assigned.

    Args:
        profile: The profile dictionary to update.

    Returns:
        A new dictionary representing the updated profile with a 'profile_id' key.
    """
    updated: Dict[str, Any] = dict(profile)  # shallow copy to avoid mutating input
    pid = updated.get("profile_id")

    if not isinstance(pid, str) or not pid or pid in existing_profile_ids:
        pid = _generate_unique_profile_id()
        updated["profile_id"] = pid

    existing_profile_ids.append(pid)
    return updated


def cache_profile_data(profile_id: str, profile_data: Dict[str, Any]) -> bool:
    """
    Cache user-specific data for rapid retrieval.

    Args:
        profile_id: The identifier for the profile to cache.
        profile_data: The data to cache for the profile.

    Returns:
        True if the data was cached successfully, False otherwise.
    """
    try:
        # Shallow copy to avoid external mutations affecting the cache
        user_cache[profile_id] = dict(profile_data)
        return True
    except Exception:
        return False
