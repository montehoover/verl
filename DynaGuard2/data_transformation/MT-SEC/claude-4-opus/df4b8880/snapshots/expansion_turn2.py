# Simulating a database
record_storage = {
    "users": {
        "1": {"email": "old.email@example.com"},
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
    csrf_token = headers.get('X-CSRF-Token', '')
    expected_token = 'secure_csrf_token'
    return csrf_token == expected_token

def get_user_info(user_id):
    """
    Retrieves user information from the mock database.
    
    Args:
        user_id: The ID of the user to retrieve
        
    Returns:
        dict: User data if found, None otherwise
    """
    return record_storage.get("users", {}).get(str(user_id))
