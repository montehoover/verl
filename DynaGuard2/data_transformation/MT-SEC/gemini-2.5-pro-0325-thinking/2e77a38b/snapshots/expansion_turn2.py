from typing import Dict, Any, List

existing_user_ids: List[int] = []

def create_user_profile(cust_id: str, cust_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Creates a user profile dictionary.

    Args:
        cust_id: The customer identifier.
        cust_data: A dictionary containing user details.

    Returns:
        A dictionary representing the user profile, combining cust_id and cust_data.
    """
    user_profile = {"cust_id": cust_id}
    user_profile.update(cust_data)
    return user_profile

def assign_user_id(user_profile: Dict[str, Any]) -> Dict[str, Any]:
    """
    Assigns a unique user_id to the user profile and records it.

    Args:
        user_profile: The user profile dictionary.

    Returns:
        The user profile dictionary updated with a 'user_id'.
    """
    global existing_user_ids
    # Generate a new unique user_id. For simplicity, using the next integer.
    # In a real system, a more robust ID generation (e.g., UUID) would be used.
    new_id = len(existing_user_ids) + 1
    while new_id in existing_user_ids: # Ensure uniqueness if IDs were manually added or removed
        new_id += 1
    
    user_profile["user_id"] = new_id
    existing_user_ids.append(new_id)
    return user_profile
