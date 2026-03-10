from typing import Dict, Any

def create_user_profile(client_id: str, info_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Creates a user profile by combining client ID and user information.
    
    Args:
        client_id: A string identifier for the client
        info_data: A dictionary containing user details
        
    Returns:
        A dictionary containing the complete user profile
    """
    user_profile = {
        'client_id': client_id,
        **info_data
    }
    return user_profile

def update_user_info(user_profile: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    """
    Updates a user profile with new information.
    
    Args:
        user_profile: The existing user profile dictionary
        updates: A dictionary containing the updates to apply
        
    Returns:
        An updated user profile dictionary
    """
    updated_profile = user_profile.copy()
    updated_profile.update(updates)
    return updated_profile
