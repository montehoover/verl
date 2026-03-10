# Global dictionary to store user information
user_storage = {}

# Global dictionary serving as the storage container for cached user information
user_cache = {}

def store_user_info(user_id, user_info):
    """
    Store user information in a global dictionary.
    
    Args:
        user_id (str): The user's ID
        user_info (str): The user's information
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        user_storage[user_id] = user_info
        return True
    except:
        return False

def cache_client_data(client_id, client_data):
    """
    Cache client data in memory for rapid retrieval.
    
    Args:
        client_id (str): The client's ID
        client_data (dict): The client's data to cache
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        user_cache[client_id] = client_data
        return True
    except:
        return False
