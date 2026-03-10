from typing import Dict, Any

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
