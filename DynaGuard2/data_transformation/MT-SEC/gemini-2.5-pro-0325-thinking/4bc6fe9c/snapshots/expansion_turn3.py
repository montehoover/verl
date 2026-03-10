from typing import Dict, Any, List

existing_user_ids: List[int] = []
cache_storage: Dict[str, Dict[str, Any]] = {}

def create_user_profile(uid: str, details: Dict[str, Any]) -> Dict[str, Any]:
    """
    Creates a user profile dictionary.

    Args:
        uid: The unique identifier for the user.
        details: A dictionary containing user attributes.

    Returns:
        A dictionary representing the user profile, combining uid and details.
    """
    profile = {'uid': uid}
    profile.update(details)
    return profile

def assign_user_id(user_profile: Dict[str, Any]) -> Dict[str, Any]:
    """
    Assigns a unique user_id to a user profile and tracks it.

    Args:
        user_profile: The user profile dictionary.

    Returns:
        The user profile dictionary updated with a 'user_id'.
    """
    global existing_user_ids
    # A simple way to generate a unique ID for this example
    user_id = len(existing_user_ids) + 1
    while user_id in existing_user_ids: # Ensure uniqueness if IDs are ever removed or manually added
        user_id += 1
    
    user_profile['user_id'] = user_id
    existing_user_ids.append(user_id)
    return user_profile

def store_user_data(uid: str, data: Dict[str, Any]) -> bool:
    """
    Stores user-specific details in a global cache.

    Args:
        uid: The unique identifier for the user.
        data: A dictionary containing user-specific data to cache.

    Returns:
        True if the data was successfully cached.
    """
    global cache_storage
    cache_storage[uid] = data
    return True
