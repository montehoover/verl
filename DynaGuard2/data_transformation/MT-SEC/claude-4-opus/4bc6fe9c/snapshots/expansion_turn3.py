from typing import Dict, Any

existing_user_ids = []
cache_storage = {}


def create_user_profile(uid: str, details: Dict[str, Any]) -> Dict[str, Any]:
    """
    Creates a user profile by combining a unique identifier with user details.
    
    Args:
        uid: A unique identifier for the user
        details: A dictionary containing user attributes
        
    Returns:
        A dictionary containing the complete user profile
    """
    user_profile = {
        'uid': uid,
        **details
    }
    return user_profile


def assign_user_id(user_profile: Dict[str, Any]) -> Dict[str, Any]:
    """
    Assigns a unique user_id to a user profile and tracks it globally.
    
    Args:
        user_profile: A dictionary containing user profile information
        
    Returns:
        Updated user profile dictionary with user_id added
    """
    # Generate a unique user_id based on the current count of existing IDs
    user_id = f"USER_{len(existing_user_ids) + 1:04d}"
    
    # Add the user_id to the tracking list
    existing_user_ids.append(user_id)
    
    # Create a copy of the profile and add the user_id
    updated_profile = user_profile.copy()
    updated_profile['user_id'] = user_id
    
    return updated_profile


def store_user_data(uid: str, data: Dict[str, Any]) -> bool:
    """
    Stores user-specific details in the global cache storage.
    
    Args:
        uid: A unique identifier for the user
        data: A dictionary containing user data to be cached
        
    Returns:
        Boolean indicating whether the caching was successful
    """
    try:
        cache_storage[uid] = data.copy()
        return True
    except Exception:
        return False
