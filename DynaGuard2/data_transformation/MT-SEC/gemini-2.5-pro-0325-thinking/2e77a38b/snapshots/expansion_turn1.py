from typing import Dict, Any

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
