# Global dictionary to store user information
user_storage = {}

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
