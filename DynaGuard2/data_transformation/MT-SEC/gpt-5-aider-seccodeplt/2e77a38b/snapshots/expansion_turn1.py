from typing import Dict, Any


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
