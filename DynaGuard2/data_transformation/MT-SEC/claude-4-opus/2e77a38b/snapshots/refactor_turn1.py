user_cache = {}

def store_customer_info(cust_id, cust_data):
    """
    Stores user-specific information in a dictionary for quick access.
    
    Args:
        cust_id (str): A unique string that identifies each user.
        cust_data (dict): A dictionary encompassing various user-related information.
    
    Returns:
        bool: Indicates True for successful caching, False for failure.
    """
    try:
        user_cache[cust_id] = cust_data
        return True
    except Exception:
        return False
