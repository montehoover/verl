# Simulating a database
database = {
    "users": {
        "1": {"phone": "123-456-7890"},
    }
}

def check_user_exists(user_id):
    """Check if a user exists in the database.
    
    Args:
        user_id (str): The user ID to check
        
    Returns:
        bool: True if user exists, False otherwise
    """
    return user_id in database.get("users", {})
