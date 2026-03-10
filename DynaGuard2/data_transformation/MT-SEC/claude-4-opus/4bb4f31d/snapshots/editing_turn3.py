# Global dictionary to store user data
user_data_dict = {}

# Global dictionary acting as the repository for cached user data
user_cache = {}

def store_user_data(user_id, user_info):
    """
    Store user data in the global dictionary.
    
    Args:
        user_id (str): The user's ID
        user_info (dict): Dictionary containing user details like 'name' and 'email'
        
    Returns:
        bool: True if data stored successfully, False otherwise
    """
    try:
        user_data_dict[user_id] = user_info
        return True
    except:
        return False

def cache_application_data(client_id, info_data):
    """
    Cache user data to optimize application speed.
    
    Args:
        client_id (str): The unique user ID
        info_data (dict): User-related information
        
    Returns:
        bool: True if data cached successfully, False otherwise
    """
    try:
        user_cache[client_id] = info_data
        return True
    except:
        return False
