def validate_csrf_token(headers):
    """
    Validates CSRF token from HTTP request headers.
    
    Args:
        headers (dict): Dictionary containing HTTP request headers
        
    Returns:
        bool: True if CSRF token is valid, False otherwise
    """
    csrf_token = headers.get('X-CSRF-Token') or headers.get('x-csrf-token')
    return csrf_token == 'secure_csrf_token'


# Simulating a database
database = {
    "users": {
        "1": {"email": "user1@example.com"},
        "2": {"email": "user2@example.com"},
        "3": {"email": "user3@example.com"},
        "4": {"email": "user4@example.com"},
    }
}


def get_user_info(user_id):
    """
    Retrieves user information from the database.
    
    Args:
        user_id: The ID of the user to retrieve
        
    Returns:
        dict: User information if found, None otherwise
    """
    return database.get("users", {}).get(str(user_id))
