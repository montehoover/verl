from typing import Dict, Any
import uuid


existing_user_ids: list[str] = []
user_cache: Dict[str, Any] = {}


def create_user_profile(cust_id: str, cust_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create a user profile dict from a customer identifier and user details.

    Args:
        cust_id: Unique identifier for the customer.
        cust_data: Dictionary of user details.

    Returns:
        A new dictionary combining cust_data with the cust_id under the "cust_id" key.
        The explicit cust_id parameter takes precedence over any "cust_id" present in cust_data.
    """
    if not isinstance(cust_id, str):
        raise TypeError("cust_id must be a string")
    if not isinstance(cust_data, dict):
        raise TypeError("cust_data must be a dict")

    profile: Dict[str, Any] = dict(cust_data)  # shallow copy to avoid mutating the input
    profile["cust_id"] = cust_id
    return profile


def _generate_unique_user_id() -> str:
    """Generate a unique user ID not present in existing_user_ids."""
    while True:
        candidate = uuid.uuid4().hex
        if candidate not in existing_user_ids:
            return candidate


def assign_user_id(user: Dict[str, Any]) -> Dict[str, Any]:
    """
    Ensure a user dictionary has a unique 'user_id' and record it globally.

    - If the user already has a 'user_id' and it's unused, it will be recorded as-is.
    - If the existing 'user_id' is already taken or invalid, a new unique one is assigned.
    - If no 'user_id' is present, a new unique one is assigned.

    Args:
        user: A dictionary representing the user.

    Returns:
        A new dictionary with a unique 'user_id' set.
    """
    if not isinstance(user, dict):
        raise TypeError("user must be a dict")

    updated: Dict[str, Any] = dict(user)  # shallow copy to avoid mutating input
    current_id = updated.get("user_id")

    if isinstance(current_id, str) and current_id:
        if current_id in existing_user_ids:
            # Collision: assign a new unique id
            new_id = _generate_unique_user_id()
            updated["user_id"] = new_id
            existing_user_ids.append(new_id)
        else:
            # Existing ID is unique; record it
            existing_user_ids.append(current_id)
    else:
        # No valid ID present; assign a new unique one
        new_id = _generate_unique_user_id()
        updated["user_id"] = new_id
        existing_user_ids.append(new_id)

    return updated


def store_customer_info(cust_id: str, cust_data: Dict[str, Any]) -> bool:
    """
    Store user-specific information in the global user_cache for quick access.

    Args:
        cust_id: The customer identifier.
        cust_data: The user data to cache.

    Returns:
        True if the data was stored successfully.
    """
    if not isinstance(cust_id, str):
        raise TypeError("cust_id must be a string")
    if not isinstance(cust_data, dict):
        raise TypeError("cust_data must be a dict")

    user_cache[cust_id] = dict(cust_data)  # store a shallow copy to avoid external mutation
    return True
