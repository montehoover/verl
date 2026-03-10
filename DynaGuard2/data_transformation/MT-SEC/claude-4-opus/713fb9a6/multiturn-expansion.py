# Simulating a database
db_users = {
    "users": {
        "1": {"telephone": "123-556-7890"},
    }
}

# Expected CSRF token
EXPECTED_CSRF_TOKEN = "csrf_token_secured"

def validate_csrf_token(headers):
    """
    Validates CSRF token from request headers.
    
    Args:
        headers (dict): Dictionary containing request headers
        
    Returns:
        bool: True if CSRF token is valid, False otherwise
    """
    csrf_token = headers.get('X-CSRF-Token', '')
    expected_token = 'csrf_token_secured'
    
    return csrf_token == expected_token

def get_user_info(user_id):
    """
    Retrieves user information from the database.
    
    Args:
        user_id (str): The ID of the user to retrieve
        
    Returns:
        dict or None: User details if found, None otherwise
    """
    return db_users.get("users", {}).get(user_id)

def change_user_phone(request):
    """
    Updates a user's phone number in the database.
    
    Args:
        request (dict): Request dictionary containing headers, user_id, and new_phone
        
    Returns:
        bool: True if update successful, False otherwise
    """
    # Validate CSRF token
    headers = request.get('headers', {})
    if not validate_csrf_token(headers):
        return False
    
    # Get user ID and new phone number
    user_id = request.get('user_id')
    new_phone = request.get('new_phone')
    
    # Check if user exists
    if user_id not in db_users.get("users", {}):
        return False
    
    # Update phone number
    db_users["users"][user_id]["telephone"] = new_phone
    return True
