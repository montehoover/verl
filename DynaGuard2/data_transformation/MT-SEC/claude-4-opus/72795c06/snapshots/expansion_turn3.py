from typing import Dict, Any
import uuid

existing_profile_ids = []
user_cache = {}

def create_user_profile(profile_id: str, profile_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Creates a user profile by combining profile ID and profile data.
    
    Args:
        profile_id: Unique identifier for the user profile
        profile_data: Dictionary containing user details
        
    Returns:
        Dictionary containing the complete user profile
    """
    profile = {
        'profile_id': profile_id,
        **profile_data
    }
    return profile

def assign_profile_id(profile: Dict[str, Any]) -> Dict[str, Any]:
    """
    Assigns a unique profile_id to a profile dictionary and tracks it globally.
    
    Args:
        profile: Dictionary containing profile data
        
    Returns:
        Updated dictionary with a unique profile_id
    """
    profile_id = str(uuid.uuid4())
    existing_profile_ids.append(profile_id)
    profile['profile_id'] = profile_id
    return profile

def cache_profile_data(profile_id: str, profile_data: dict) -> bool:
    try:
        user_cache[profile_id] = profile_data
        return True
    except Exception as e:
        return False
