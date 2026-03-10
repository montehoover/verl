from typing import Dict, Any


def create_user_profile(client_id: str, info_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create a local user profile by combining the client_id with the provided info_data.

    The returned dictionary includes:
    - all key/value pairs from info_data
    - a "client_id" field set to the provided client_id (overrides any existing key)
    """
    if not isinstance(client_id, str):
        raise TypeError("client_id must be a string")
    if not isinstance(info_data, dict):
        raise TypeError("info_data must be a dictionary")

    # Merge info_data with client_id, ensuring client_id takes precedence
    profile: Dict[str, Any] = {**info_data, "client_id": client_id}
    return profile
