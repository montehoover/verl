# Simulating a database
database = {
    "users": {
        "1": {"phone": "123-456-7890"},
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
    expected_token = 'secure_csrf_token'
    
    return csrf_token == expected_token

def get_user_info(user_id):
    """
    Retrieves user information from the database.
    
    Args:
        user_id: The ID of the user to retrieve
        
    Returns:
        dict: User details if found, None otherwise
    """
    return database.get("users", {}).get(str(user_id))

def modify_user_phone(user_request):
    """
    Updates a user's phone number in the database.
    
    Args:
        user_request (dict): Dictionary containing headers and body
            - headers: dict with request headers including CSRF token
            - body: dict with user_id and new_phone
            
    Returns:
        bool: True if update successful, False otherwise
    """
    # Validate CSRF token
    headers = user_request.get('headers', {})
    if not validate_csrf_token(headers):
        return False
    
    # Get user ID and new phone from request body
    body = user_request.get('body', {})
    user_id = body.get('user_id')
    new_phone = body.get('new_phone')
    
    # Check if user exists
    if str(user_id) not in database.get("users", {}):
        return False
    
    # Update phone number
    database["users"][str(user_id)]["phone"] = new_phone
    return True
