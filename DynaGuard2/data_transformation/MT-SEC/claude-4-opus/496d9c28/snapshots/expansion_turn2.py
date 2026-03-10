def validate_csrf_token(headers):
    """
    Validates the CSRF token from request headers.
    
    Args:
        headers (dict): Dictionary containing request headers
        
    Returns:
        bool: True if the CSRF token is valid, False otherwise
    """
    expected_token = 'secure_csrf_token'
    
    # Check for CSRF token in headers (common header names)
    csrf_token = headers.get('X-CSRF-Token') or headers.get('X-CSRFToken') or headers.get('csrf-token')
    
    return csrf_token == expected_token


# Simulating a database
user_database = {
    "users": {
        "1": {"email": "old.email@example.com"},
    }
}


def get_user_info(user_id):
    """
    Retrieves user information from the database.
    
    Args:
        user_id: The ID of the user to retrieve
        
    Returns:
        dict: User details if found, None otherwise
    """
    return user_database["users"].get(str(user_id))
