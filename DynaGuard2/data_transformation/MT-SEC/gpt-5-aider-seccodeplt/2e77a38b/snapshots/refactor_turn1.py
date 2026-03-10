user_cache = {}

def store_customer_info(cust_id: str, cust_data: dict) -> bool:
    """
    Stores user-specific information in a global in-memory cache.

    Args:
        cust_id (str): A unique string that identifies each user.
        cust_data (dict): A dictionary encompassing various user-related information.

    Returns:
        bool: True for successful caching, False for failure.
    """
    global user_cache
    try:
        if not isinstance(cust_id, str) or not cust_id:
            return False
        if not isinstance(cust_data, dict):
            return False

        # Store a shallow copy to avoid external mutations affecting the cache.
        user_cache[cust_id] = dict(cust_data)
        return True
    except Exception:
        return False
