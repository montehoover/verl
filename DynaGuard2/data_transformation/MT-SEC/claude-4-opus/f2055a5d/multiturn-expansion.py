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

def remove_user_account(input_request):
    """
    Removes a user account from the database after verifying CSRF token.
    
    Args:
        input_request (dict): Dictionary containing request data with headers and user_id
        
    Returns:
        bool: True if deletion is successful, False otherwise
    """
    # Verify CSRF token
    headers = input_request.get('headers', {})
    if not validate_request(headers):
        return False
    
    # Get user ID from request
    user_id = input_request.get('user_id')
    if user_id is None:
        return False
    
    # Check if user exists and remove
    user_id_str = str(user_id)
    if user_id_str in database.get("users", {}):
        del database["users"][user_id_str]
        return True
    
    return False
