# Global dictionary to store user data
user_data_dict = {}

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
