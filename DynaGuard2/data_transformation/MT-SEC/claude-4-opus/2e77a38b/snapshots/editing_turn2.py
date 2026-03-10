# Global dictionary to store users
users = {}

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
