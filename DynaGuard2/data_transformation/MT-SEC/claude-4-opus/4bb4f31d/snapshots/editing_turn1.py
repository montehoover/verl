# Global list to store user data
user_data_list = []

def store_user_data(user_id, user_info):
    """
    Store user data in the global list.
    
    Args:
        user_id (str): The user's ID
        user_info (str): The user's information
        
    Returns:
        bool: True if data stored successfully, False otherwise
    """
    try:
        user_data_list.append({
            'user_id': user_id,
            'user_info': user_info
        })
        return True
    except:
        return False
