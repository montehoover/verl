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
    Validates the CSRF token from request headers.
    
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
    return database.get("users", {}).get(user_id)

def change_user_phone(request_details):
    """
    Updates a user's phone number in the database.
    
    Args:
        request_details (dict): Dictionary containing 'headers' and 'body' with user_id and new_phone
        
    Returns:
        bool: True if update successful, False otherwise
    """
    # Validate CSRF token
    headers = request_details.get('headers', {})
    if not validate_csrf_token(headers):
        return False
    
    # Get user_id and new_phone from body
    body = request_details.get('body', {})
    user_id = body.get('user_id')
    new_phone = body.get('new_phone')
    
    # Check if user exists
    if user_id not in database.get("users", {}):
        return False
    
    # Update phone number
    database["users"][user_id]["phone"] = new_phone
    return True
