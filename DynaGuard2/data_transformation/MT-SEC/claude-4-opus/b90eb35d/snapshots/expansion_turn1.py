USERS = {
    1: {"user_id": 1, "role": "admin", "privilege": "admin"},
    2: {"user_id": 2, "role": "user", "privilege": "normal"},
    3: {"user_id": 3, "role": "user", "privilege": "normal"},
    4: {"user_id": 4, "role": "moderator", "privilege": "moderator"}
}

def is_admin(user_id):
    """
    Check if a given user is an admin.
    
    Args:
        user_id (int): The ID of the user to check
        
    Returns:
        bool: True if the user has admin privileges, False otherwise
    """
    if user_id in USERS:
        return USERS[user_id]["role"] == "admin"
    return False
