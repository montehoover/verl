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


def change_user_contact(transaction_request):
    """
    Updates a user's phone number in the database.
    
    Args:
        transaction_request (dict): Dictionary containing headers and body with user ID and new phone number
        
    Returns:
        bool: True if update successful, False otherwise
    """
    # Validate CSRF token
    headers = transaction_request.get('headers', {})
    if not validate_request(headers):
        return False
    
    # Extract user ID and new phone number from body
    body = transaction_request.get('body', {})
    user_id = body.get('user_id')
    new_phone = body.get('phone')
    
    # Check if user exists and update phone number
    if user_id in database.get("users", {}):
        database["users"][user_id]["phone"] = new_phone
        return True
    
    return False
