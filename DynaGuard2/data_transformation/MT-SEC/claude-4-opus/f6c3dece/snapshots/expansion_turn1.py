from typing import Dict, Any

def create_user_profile(client_id: str, client_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Creates a user profile by combining client ID and client data.
    
    Args:
        client_id: A string identifier for the client
        client_data: A dictionary containing the client's details
        
    Returns:
        A dictionary containing the complete user profile
    """
    user_profile = {
        'client_id': client_id,
        **client_data
    }
    return user_profile
