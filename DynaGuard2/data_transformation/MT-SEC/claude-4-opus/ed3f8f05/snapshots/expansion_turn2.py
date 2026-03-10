# Simulating a database
database = {
    "users": {
        "1": {"email": "old.email@example.com"},
    }
}

def verify_csrf_token(headers):
    """
    Verify CSRF token in request headers.
    
    Args:
        headers (dict): Request headers dictionary
        
    Returns:
        bool: True if token matches 'secure_csrf_token', False otherwise
    """
    csrf_token = headers.get('X-CSRF-Token', '')
    return csrf_token == 'secure_csrf_token'

def get_user_information(user_id):
    """
    Retrieve user information from database.
    
    Args:
        user_id: The ID of the user to retrieve
        
    Returns:
        dict: User data if found, None otherwise
    """
    return database.get("users", {}).get(str(user_id))
