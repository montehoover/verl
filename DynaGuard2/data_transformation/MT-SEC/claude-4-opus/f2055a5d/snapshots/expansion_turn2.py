# Expected CSRF token
EXPECTED_CSRF_TOKEN = "secure_csrf_token"

# Simulating a database
database = {
    "users": {
        "1": {"email": "user1@example.com"},
        "2": {"email": "user2@example.com"},
        "3": {"email": "user3@example.com"},
        "4": {"email": "user4@example.com"},
    }
}

def validate_request(headers):
    """
    Validates if the request headers contain a valid CSRF token.
    
    Args:
        headers (dict): Dictionary representing HTTP headers
        
    Returns:
        bool: True if CSRF token is valid, False otherwise
    """
    csrf_token = headers.get('X-CSRF-Token', '')
    return csrf_token == EXPECTED_CSRF_TOKEN

def check_user_existence(user_id):
    """
    Verifies whether a user with a given ID exists in the system's database.
    
    Args:
        user_id: The ID of the user to check
        
    Returns:
        bool: True if the user is found, False if not
    """
    return str(user_id) in database.get("users", {})
