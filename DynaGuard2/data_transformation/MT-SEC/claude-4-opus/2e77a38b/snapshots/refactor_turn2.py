# Global cache for storing user data
user_cache = {}


def store_customer_info(cust_id, cust_data):
    """
    Stores user-specific information in a dictionary for quick access.
    
    This function implements a memory-based caching system to optimize
    application performance by storing user data in memory, eliminating
    the need for repeated database queries.
    
    Args:
        cust_id (str): A unique string that identifies each user.
        cust_data (dict): A dictionary encompassing various user-related 
                          information.
    
    Returns:
        bool: True if the data was successfully cached, False if an error
              occurred during the caching process.
    
    Example:
        >>> user_data = {'name': 'John Doe', 'email': 'john@example.com'}
        >>> store_customer_info('user123', user_data)
        True
    """
    try:
        # Store the customer data in the global cache using the customer ID as key
        user_cache[cust_id] = cust_data
        
        # Return True to indicate successful caching
        return True
        
    except Exception:
        # Return False if any exception occurs during the caching process
        return False
