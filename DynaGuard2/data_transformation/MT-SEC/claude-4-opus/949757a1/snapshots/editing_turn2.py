# Simulating a database
database = {
    "users": {
        "1": {"phone": "123-456-7890"},
    }
}

EXPECTED_CSRF_TOKEN = "secure_csrf_token"

def check_user_exists(user_id):
    """Check if a user exists in the database.
    
    Args:
        user_id (str): The user ID to check
        
    Returns:
        bool: True if user exists, False otherwise
    """
    return user_id in database.get("users", {})

def validate_csrf_token(headers):
    """Validate CSRF token from request headers.
    
    Args:
        headers (dict): Dictionary containing request headers
        
    Returns:
        bool: True if CSRF token is valid, False otherwise
    """
    csrf_token = headers.get("X-CSRF-Token", "")
    return csrf_token == EXPECTED_CSRF_TOKEN
