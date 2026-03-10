# Simulating a database
database = {
    "users": {
        "1": {"phone": "123-456-7890"},
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
    return csrf_token == 'secure_csrf_token'

def get_user_info(user_id):
    """
    Retrieves user information from the database.
    
    Args:
        user_id (str): The ID of the user to retrieve
        
    Returns:
        dict or None: User details if found, None otherwise
    """
    return database.get("users", {}).get(user_id)
