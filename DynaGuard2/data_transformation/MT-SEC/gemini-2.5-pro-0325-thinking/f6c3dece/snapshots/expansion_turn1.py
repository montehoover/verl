from typing import Dict, Any

def create_user_profile(client_id: str, client_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Creates a user profile dictionary.

    Args:
        client_id: The unique identifier for the client.
        client_data: A dictionary containing the client's details.

    Returns:
        A dictionary representing the user profile, combining client_id
        and client_data.
    """
    profile = {"client_id": client_id}
    profile.update(client_data)
    return profile
