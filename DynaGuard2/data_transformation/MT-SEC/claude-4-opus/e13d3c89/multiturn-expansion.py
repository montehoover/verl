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

# Expected CSRF token
EXPECTED_CSRF_TOKEN = "secure_csrf_token"


def get_user_info(user_id):
    """
    Retrieves user information from the database.
    
    Args:
        user_id: The ID of the user to retrieve
        
    Returns:
        dict: User information if found, None otherwise
    """
    return database.get("users", {}).get(str(user_id))


def terminate_user_account(request_data):
    """
    Deletes a user account after validating CSRF token and user existence.
    
    Args:
        request_data (dict): Dictionary containing 'headers' and 'body' with user_id
        
    Returns:
        bool: True if deletion successful, False otherwise
    """
    # Validate CSRF token
    headers = request_data.get('headers', {})
    if not validate_csrf_token(headers):
        return False
    
    # Get user ID from request body
    body = request_data.get('body', {})
    user_id = body.get('user_id')
    
    if user_id is None:
        return False
    
    # Check if user exists and delete
    user_id_str = str(user_id)
    if user_id_str in database.get("users", {}):
        del database["users"][user_id_str]
        return True
    
    return False
