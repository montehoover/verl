def authenticate_user(user_id: str, password: str) -> bool:
    """
    Authenticates a user based on their user_id and password.
    
    Args:
        user_id (str): The user's unique identifier
        password (str): The user's password
        
    Returns:
        bool: True if authentication successful, False otherwise
    """
    # This is a placeholder implementation
    # In production, this would check against a secure database
    # with properly hashed passwords
    
    # Example hardcoded users for demonstration
    # Never store passwords in plain text in production!
    valid_users = {
        "user123": "securepass456",
        "admin": "adminpass789",
        "john_doe": "password123"
    }
    
    # Check if user exists and password matches
    if user_id in valid_users and valid_users[user_id] == password:
        return True
    
    return False
