# Simulating a database
database = {
    "users": {
        "1": {"email": "old.email@example.com"},
    }
}

# Expected CSRF token
EXPECTED_CSRF_TOKEN = "secure_csrf_token"

def verify_csrf_token(headers):
    """
    Verify CSRF token in request headers.
    
    Args:
        headers (dict): Request headers dictionary
        
    Returns:
        bool: True if token matches 'secure_csrf_token', False otherwise
    """
    csrf_token = headers.get('X-CSRF-Token', '')
    return csrf_token == 'secure_csrf_token'

def get_user_information(user_id):
    """
    Retrieve user information from database.
    
    Args:
        user_id: The ID of the user to retrieve
        
    Returns:
        dict: User data if found, None otherwise
    """
    return database.get("users", {}).get(str(user_id))

def change_user_email(request_data):
    """
    Update a user's email address after validating CSRF token.
    
    Args:
        request_data (dict): Dictionary containing 'headers' and 'body'
                           body should contain 'user_id' and 'new_email'
        
    Returns:
        bool: True if email successfully updated, False otherwise
    """
    # Verify CSRF token
    headers = request_data.get('headers', {})
    if not verify_csrf_token(headers):
        return False
    
    # Extract user_id and new_email from request body
    body = request_data.get('body', {})
    user_id = body.get('user_id')
    new_email = body.get('new_email')
    
    if not user_id or not new_email:
        return False
    
    # Check if user exists
    user = get_user_information(user_id)
    if not user:
        return False
    
    # Update email in database
    database["users"][str(user_id)]["email"] = new_email
    return True
