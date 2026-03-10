# Global dictionary to store users
users = {}

# Global dictionary acting as the repository for cached user data
user_cache = {}

def add_user(user_id, user_info):
    """
    Add a user to the global users dictionary.
    
    Args:
        user_id (str): The user's ID
        user_info (dict): The user's information containing details like 'name' and 'email'
        
    Returns:
        bool: True if user was added successfully, False otherwise
    """
    try:
        # Add user to the dictionary with user_id as key
        users[user_id] = user_info
        return True
    except:
        return False

def store_customer_info(cust_id, cust_data):
    """
    Store customer information in the user cache.
    
    Args:
        cust_id (str): The unique customer ID
        cust_data (dict): Customer-related information
        
    Returns:
        bool: True if customer info was stored successfully, False otherwise
    """
    try:
        # Store customer data in the cache with cust_id as key
        user_cache[cust_id] = cust_data
        return True
    except:
        return False
