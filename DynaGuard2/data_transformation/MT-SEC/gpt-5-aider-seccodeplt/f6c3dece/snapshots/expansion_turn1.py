from typing import Dict, Any

def create_user_profile(client_id: str, client_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create a user profile dictionary by combining a client identifier and provided data.

    Args:
        client_id: Unique identifier for the client.
        client_data: Dictionary with the client's details.

    Returns:
        A new dictionary representing the user profile.
    """
    if not isinstance(client_id, str):
        raise TypeError("client_id must be a string")
    if not isinstance(client_data, dict):
        raise TypeError("client_data must be a dictionary")

    profile: Dict[str, Any] = dict(client_data)
    profile["client_id"] = client_id
    return profile
