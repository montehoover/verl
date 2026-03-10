# Simulating a database
database = {
    "users": {
        "1": {"phone": "123-456-7890"},
    }
}

def get_user_phone(user_id):
    """Get a user's phone number from the database.
    
    Args:
        user_id (str): The ID of the user to check
        
    Returns:
        str or None: The user's phone number if they exist, None otherwise
    """
    users = database.get("users", {})
    if user_id in users:
        return users[user_id].get("phone")
    return None
