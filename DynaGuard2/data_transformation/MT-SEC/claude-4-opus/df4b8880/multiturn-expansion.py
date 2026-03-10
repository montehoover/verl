# Simulating a database
record_storage = {
    "users": {
        "1": {"email": "old.email@example.com"},
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
    Retrieves user information from the mock database.
    
    Args:
        user_id: The ID of the user to retrieve
        
    Returns:
        dict: User data if found, None otherwise
    """
    return record_storage.get("users", {}).get(str(user_id))

def alter_user_email(request):
    """
    Updates a user's email address in the mock database.
    
    Args:
        request (dict): Request dictionary containing headers, user_id, and new_email
        
    Returns:
        bool: True if update successful, False otherwise
    """
    # Validate CSRF token
    headers = request.get('headers', {})
    if not validate_csrf_token(headers):
        return False
    
    # Get user ID and new email from request
    user_id = request.get('user_id')
    new_email = request.get('new_email')
    
    if not user_id or not new_email:
        return False
    
    # Check if user exists
    if str(user_id) not in record_storage.get("users", {}):
        return False
    
    # Update user's email
    record_storage["users"][str(user_id)]["email"] = new_email
    return True
