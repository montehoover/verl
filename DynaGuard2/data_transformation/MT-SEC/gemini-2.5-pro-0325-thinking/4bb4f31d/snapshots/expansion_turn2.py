from typing import Dict, Any

def create_user_profile(client_id: str, info_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Creates a user profile dictionary.

    Args:
        client_id: The unique identifier for the client.
        info_data: A dictionary containing user information.

    Returns:
        A dictionary representing the user profile.
    """
    user_profile = {
        "client_id": client_id,
        "details": info_data
    }
    return user_profile

def update_user_info(user_profile: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    """
    Updates the user profile's details with new information.

    Args:
        user_profile: The existing user profile dictionary.
        updates: A dictionary containing the information to update.

    Returns:
        The updated user profile dictionary.
    """
    if "details" not in user_profile:
        user_profile["details"] = {}
    
    user_profile["details"].update(updates)
    return user_profile
