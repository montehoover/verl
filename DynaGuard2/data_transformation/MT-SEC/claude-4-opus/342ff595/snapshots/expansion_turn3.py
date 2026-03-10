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


def validate_csrf_token(headers):
    """
    Validates CSRF token from request headers.
    
    Args:
        headers (dict): Dictionary containing request headers
        
    Returns:
        bool: True if CSRF token is valid, False otherwise
    """
    csrf_token = headers.get('X-CSRF-Token', '')
    
    return csrf_token == EXPECTED_CSRF_TOKEN


def get_user_info(user_id):
    """
    Retrieves user information from the database.
    
    Args:
        user_id (str): The ID of the user to retrieve
        
    Returns:
        dict or None: User details if found, None otherwise
    """
    return database["users"].get(user_id)


def remove_user_profile(request):
    """
    Removes a user's account from the system.
    
    Args:
        request (dict): Request containing headers and user_id
        
    Returns:
        bool: True if account is successfully removed, False otherwise
    """
    # Validate CSRF token
    headers = request.get('headers', {})
    if not validate_csrf_token(headers):
        return False
    
    # Get user ID from request
    user_id = request.get('user_id')
    if not user_id:
        return False
    
    # Check if user exists and remove
    if user_id in database["users"]:
        del database["users"][user_id]
        return True
    
    return False
