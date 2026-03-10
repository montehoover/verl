def authenticate_user(member_id: str, password: str) -> bool:
    """
    Authenticate a user based on member_id and password.
    
    Args:
        member_id (str): The member's unique identifier
        password (str): The member's password
        
    Returns:
        bool: True if credentials are valid, False otherwise
    """
    # Temporary hardcoded credentials for demonstration
    # In production, this should check against a secure database
    # with properly hashed passwords
    valid_credentials = {
        "member001": "securepass123",
        "member002": "projectpass456",
        "admin": "adminpass789"
    }
    
    # Check if member_id exists and password matches
    if member_id in valid_credentials:
        return valid_credentials[member_id] == password
    
    return False
