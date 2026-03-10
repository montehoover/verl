from typing import Dict, Any, List
import uuid

existing_user_ids: List[str] = []

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

def assign_user_id(user_profile: Dict[str, Any]) -> Dict[str, Any]:
    """
    Assigns a unique user_id to the user profile and records it.

    Args:
        user_profile: The user profile dictionary.

    Returns:
        The user profile dictionary updated with a 'user_id'.
    """
    global existing_user_ids
    # Generate a unique ID. Using UUID for robustness.
    # A simpler approach could be f"{user_profile.get('client_id', 'unknown')}_{len(existing_user_ids)}"
    # but UUID is generally better for uniqueness.
    user_id = str(uuid.uuid4())
    
    # Ensure the generated ID is truly unique if somehow a collision occurs (highly unlikely with UUIDv4)
    while user_id in existing_user_ids:
        user_id = str(uuid.uuid4())
        
    user_profile["user_id"] = user_id
    existing_user_ids.append(user_id)
    return user_profile
