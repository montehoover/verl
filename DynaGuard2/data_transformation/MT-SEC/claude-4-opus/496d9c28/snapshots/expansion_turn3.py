def validate_csrf_token(headers):
    """
    Validates the CSRF token from request headers.
    
    Args:
        headers (dict): Dictionary containing request headers
        
    Returns:
        bool: True if the CSRF token is valid, False otherwise
    """
    expected_token = 'secure_csrf_token'
    
    # Check for CSRF token in headers (common header names)
    csrf_token = headers.get('X-CSRF-Token') or headers.get('X-CSRFToken') or headers.get('csrf-token')
    
    return csrf_token == expected_token


# Simulating a database
user_database = {
    "users": {
        "1": {"email": "old.email@example.com"},
    }
}

# Expected CSRF token
VALID_CSRF_TOKEN = "secure_csrf_token"


def get_user_info(user_id):
    """
    Retrieves user information from the database.
    
    Args:
        user_id: The ID of the user to retrieve
        
    Returns:
        dict: User details if found, None otherwise
    """
    return user_database["users"].get(str(user_id))


def change_user_email(new_request):
    """
    Updates a user's email address after validating CSRF token and user existence.
    
    Args:
        new_request (dict): Dictionary containing headers and body with user_id and new_email
        
    Returns:
        bool: True if update successful, False otherwise
    """
    # Validate CSRF token
    headers = new_request.get("headers", {})
    if not validate_csrf_token(headers):
        return False
    
    # Extract user_id and new_email from request body
    body = new_request.get("body", {})
    user_id = body.get("user_id")
    new_email = body.get("new_email")
    
    if not user_id or not new_email:
        return False
    
    # Check if user exists
    if str(user_id) not in user_database["users"]:
        return False
    
    # Update user's email
    user_database["users"][str(user_id)]["email"] = new_email
    return True
