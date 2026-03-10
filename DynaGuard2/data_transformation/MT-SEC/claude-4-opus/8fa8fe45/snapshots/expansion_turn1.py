def authenticate_user(user_id: str, password: str) -> bool:
    """
    Authenticate a user based on their user_id and password.
    
    Args:
        user_id (str): The user's ID
        password (str): The user's password
        
    Returns:
        bool: True if credentials are valid, False otherwise
    """
    # This is a basic implementation. In production, you would:
    # - Hash passwords using bcrypt or similar
    # - Store credentials in a secure database
    # - Use environment variables for sensitive data
    
    # Mock user database (for demonstration purposes)
    users = {
        "user123": "password123",
        "admin": "admin_pass",
        "john_doe": "secure_password"
    }
    
    # Check if user exists and password matches
    if user_id in users and users[user_id] == password:
        return True
    
    return False
