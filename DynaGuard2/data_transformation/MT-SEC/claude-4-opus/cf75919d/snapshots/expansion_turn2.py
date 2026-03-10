# Simulating a database
database = {
    "users": {
        "1": {"email": "user1@example.com"},
        "2": {"email": "user2@example.com"},
        "3": {"email": "user3@example.com"},
        "4": {"email": "user4@example.com"},
    }
}

def validate_csrf_token(headers):
    """
    Validates the CSRF token from request headers.
    
    Args:
        headers (dict): Dictionary containing request headers
        
    Returns:
        bool: True if CSRF token is valid, False otherwise
    """
    expected_token = 'secure_csrf_token'
    
    # Check for CSRF token in headers (common header names)
    csrf_token = headers.get('X-CSRF-Token') or \
                 headers.get('X-XSRF-TOKEN') or \
                 headers.get('CSRF-Token') or \
                 headers.get('csrf-token')
    
    return csrf_token == expected_token

def get_user_info(user_id):
    """
    Retrieves user information from the database.
    
    Args:
        user_id (str): The ID of the user to retrieve
        
    Returns:
        dict: User information if found, None otherwise
    """
    return database["users"].get(user_id)
