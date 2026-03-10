# Global list to store user information
users = []

def store_user_info(user_id, user_name):
    """
    Store user information in the global users list.
    
    Args:
        user_id (str): The user's ID
        user_name (str): The user's name
        
    Returns:
        bool: True if stored successfully, False otherwise
    """
    try:
        user_info = {
            'user_id': user_id,
            'user_name': user_name
        }
        users.append(user_info)
        return True
    except:
        return False
