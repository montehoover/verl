# Simulating a database
email_database = {
    "users": {
        "1": {"email": "old.email@example.com"},
    }
}

def validate_csrf(headers):
    """
    Validates CSRF token from headers dictionary.
    
    Args:
        headers: Dictionary containing HTTP headers
        
    Returns:
        bool: True if CSRF token is valid, False otherwise
    """
    csrf_token = headers.get('X-CSRF-Token', '')
    expected_token = 'secure_csrf_token'
    
    return csrf_token == expected_token

def fetch_user_by_id(user_id):
    """
    Retrieves user information by ID from the email database.
    
    Args:
        user_id: String representing the user ID
        
    Returns:
        dict: User data if found, None otherwise
    """
    return email_database.get("users", {}).get(user_id)
