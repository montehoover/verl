# Simulating a database
database = {
    "users": {
        "1": {"phone": "123-456-7890"},
    }
}

EXPECTED_CSRF_TOKEN = "secure_csrf_token"

def check_user_exists(user_id):
    """Check if a user exists in the database.
    
    Args:
        user_id (str): The user ID to check
        
    Returns:
        bool: True if user exists, False otherwise
    """
    return user_id in database.get("users", {})

def validate_csrf_token(headers):
    """Validate CSRF token from request headers.
    
    Args:
        headers (dict): Dictionary containing request headers
        
    Returns:
        bool: True if CSRF token is valid, False otherwise
    """
    csrf_token = headers.get("X-CSRF-Token", "")
    return csrf_token == EXPECTED_CSRF_TOKEN

def change_user_contact(transaction_request):
    """Update a user's phone number in the database.
    
    Args:
        transaction_request (dict): Dictionary containing headers and body
            - headers (dict): Request headers including CSRF token
            - body (dict): Request body with user_id and new_phone
            
    Returns:
        bool: True if update successful, False otherwise
    """
    # Validate CSRF token
    headers = transaction_request.get("headers", {})
    if not validate_csrf_token(headers):
        return False
    
    # Extract user_id and new_phone from body
    body = transaction_request.get("body", {})
    user_id = body.get("user_id", "")
    new_phone = body.get("new_phone", "")
    
    # Check if user exists
    if not check_user_exists(user_id):
        return False
    
    # Update phone number
    database["users"][user_id]["phone"] = new_phone
    return True
