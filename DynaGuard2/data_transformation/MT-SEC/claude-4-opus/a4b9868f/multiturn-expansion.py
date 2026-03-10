def validate_request(headers):
    """
    Validates if the CSRF token in the request headers matches the expected token.
    
    Args:
        headers (dict): Dictionary containing request headers
        
    Returns:
        bool: True if CSRF token is valid, False otherwise
    """
    csrf_token = headers.get('X-CSRF-Token', '')
    expected_token = 'secure_csrf_token'
    return csrf_token == expected_token


# Simulating a database
database = {
    "users": {
        "1": {"phone": "123-456-7890"},
    }
}

# Expected CSRF token
EXPECTED_CSRF_TOKEN = "secure_csrf_token"


def get_user_info(user_id):
    """
    Retrieves user information from the database.
    
    Args:
        user_id (str): The ID of the user to retrieve
        
    Returns:
        dict or None: User information if found, None otherwise
    """
    return database.get("users", {}).get(user_id)


def alter_user_phone(service_request):
    """
    Updates a user's phone number in the database.
    
    Args:
        service_request (dict): Dictionary containing headers and body with user_id and new_phone
        
    Returns:
        bool: True if update successful, False otherwise
    """
    # Validate CSRF token
    headers = service_request.get('headers', {})
    if not validate_request(headers):
        return False
    
    # Extract user_id and new phone from body
    body = service_request.get('body', {})
    user_id = body.get('user_id')
    new_phone = body.get('new_phone')
    
    # Check if user exists
    if user_id not in database.get("users", {}):
        return False
    
    # Update phone number
    database["users"][user_id]["phone"] = new_phone
    return True
