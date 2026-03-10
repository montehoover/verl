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
