# Global list to store users
users = []

def add_user(user_id, user_info):
    """
    Add a user to the global users list.
    
    Args:
        user_id (str): The user's ID
        user_info (str): The user's information
        
    Returns:
        bool: True if user was added successfully, False otherwise
    """
    try:
        # Add user as a tuple (user_id, user_info)
        users.append((user_id, user_info))
        return True
    except:
        return False
