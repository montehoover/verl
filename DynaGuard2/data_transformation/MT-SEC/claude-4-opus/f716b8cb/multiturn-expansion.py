# Simulating a database
email_database = {
    "users": {
        "1": {"email": "old.email@example.com"},
    }
}

# Expected CSRF token
EXPECTED_CSRF_TOKEN = "secure_csrf_token"

def validate_csrf(headers):
    """
    Validates CSRF token from headers dictionary.
    
    Args:
        headers: Dictionary containing HTTP headers
        
    Returns:
        bool: True if CSRF token is valid, False otherwise
    """
    csrf_token = headers.get('X-CSRF-Token', '')
    expected_token = 'secure_csrf_token'
    
    return csrf_token == expected_token

def fetch_user_by_id(user_id):
    """
    Retrieves user information by ID from the email database.
    
    Args:
        user_id: String representing the user ID
        
    Returns:
        dict: User data if found, None otherwise
    """
    return email_database.get("users", {}).get(user_id)

def modify_user_email_address(user_request):
    """
    Updates a user's email address.
    
    Args:
        user_request: Dictionary containing headers and body with user ID and new email
        
    Returns:
        bool: True if update is successful, False otherwise
    """
    # Validate CSRF token
    headers = user_request.get("headers", {})
    if not validate_csrf(headers):
        return False
    
    # Extract user ID and new email from request body
    body = user_request.get("body", {})
    user_id = body.get("user_id")
    new_email = body.get("email")
    
    # Check if user exists and update email
    if user_id and new_email and user_id in email_database.get("users", {}):
        email_database["users"][user_id]["email"] = new_email
        return True
    
    return False
