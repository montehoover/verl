def authenticate_user(user_id: str, password: str) -> bool:
    """
    Authenticate a user based on user_id and password.
    
    Args:
        user_id (str): The user's ID
        password (str): The user's password
        
    Returns:
        bool: True if credentials are valid, False otherwise
    """
    # This is a simple example - in production, you would:
    # 1. Hash passwords using bcrypt or similar
    # 2. Store credentials in a secure database
    # 3. Use proper session management
    
    # Example hardcoded users for demonstration
    # In production, fetch from database and compare hashed passwords
    valid_users = {
        "user123": "password123",
        "admin": "adminpass",
        "john_doe": "securepass"
    }
    
    # Check if user exists and password matches
    if user_id in valid_users and valid_users[user_id] == password:
        return True
    
    return False
