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
