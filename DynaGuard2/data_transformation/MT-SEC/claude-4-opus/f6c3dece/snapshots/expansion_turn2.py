from typing import Dict, Any
import uuid

existing_user_ids = []

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

def assign_user_id(user_profile: Dict[str, Any]) -> Dict[str, Any]:
    """
    Assigns a unique user_id to a user profile and records it globally.
    
    Args:
        user_profile: A dictionary containing user profile information
        
    Returns:
        An updated dictionary with a unique 'user_id' added
    """
    user_id = str(uuid.uuid4())
    existing_user_ids.append(user_id)
    
    updated_profile = user_profile.copy()
    updated_profile['user_id'] = user_id
    
    return updated_profile
