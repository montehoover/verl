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
    Validates the CSRF token from HTTP headers.
    
    Args:
        headers (dict): Dictionary containing HTTP headers
        
    Returns:
        bool: True if the CSRF token is valid, False otherwise
    """
    csrf_token = headers.get('X-CSRF-Token', '')
    return csrf_token == EXPECTED_CSRF_TOKEN

def get_user_details(user_id):
    """
    Retrieves user details from the database.
    
    Args:
        user_id (str): The ID of the user to retrieve
        
    Returns:
        dict or None: User details if found, None otherwise
    """
    return database.get("users", {}).get(user_id)

def remove_account(request):
    """
    Removes a user account from the system.
    
    Args:
        request (dict): Request dictionary containing headers and user_id
        
    Returns:
        bool: True if account is successfully deleted, False otherwise
    """
    headers = request.get('headers', {})
    user_id = request.get('user_id')
    
    if not validate_csrf_token(headers):
        return False
    
    if user_id in database.get("users", {}):
        del database["users"][user_id]
        return True
    
    return False
