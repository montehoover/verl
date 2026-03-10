# In-memory cache for user data
user_cache = {}

def store_customer_info(cust_id: str, cust_data: dict) -> bool:
    """
    Stores user-specific information in a global in-memory cache for fast access.

    Args:
        cust_id (str): A unique string that identifies each user.
        cust_data (dict): A dictionary encompassing various user-related information.

    Returns:
        bool: True for successful caching, False for failure.
    """
    global user_cache

    # Validate inputs
    if not isinstance(cust_id, str) or not cust_id:
        return False
    if not isinstance(cust_data, dict):
        return False

    try:
        user_cache[cust_id] = cust_data
        return True
    except Exception:
        # In case of any unexpected error during caching
        return False
