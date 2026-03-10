def check_csrf_token(headers):
    """
    Verifies whether the CSRF token in the headers matches the expected token.
    
    Args:
        headers (dict): Dictionary containing request headers
        
    Returns:
        bool: True if the CSRF token is valid, False otherwise
    """
    csrf_token = headers.get('X-CSRF-Token', '')
    return csrf_token == 'secure_csrf_token'


# Simulating a database
account_database = {
    "users": {
        "1": {"email": "old.email@example.com"},
    }
}

# Expected CSRF token
VALID_TOKEN = "secure_csrf_token"


def get_user_by_id(user_id):
    """
    Retrieves user details from the account_database by user ID.
    
    Args:
        user_id: The ID of the user to retrieve
        
    Returns:
        dict: User data if found, None otherwise
    """
    return account_database["users"].get(str(user_id))


def edit_user_email(payload):
    """
    Updates a user's email address after verifying CSRF token and user existence.
    
    Args:
        payload (dict): Dictionary containing:
            - headers (dict): Request headers including CSRF token
            - body (dict): Request body with user_id and new_email
            
    Returns:
        bool: True if email update succeeds, False otherwise
    """
    # Verify CSRF token
    headers = payload.get('headers', {})
    if not check_csrf_token(headers):
        return False
    
    # Extract user_id and new_email from body
    body = payload.get('body', {})
    user_id = body.get('user_id')
    new_email = body.get('new_email')
    
    if user_id is None or new_email is None:
        return False
    
    # Check if user exists
    user = get_user_by_id(user_id)
    if user is None:
        return False
    
    # Update user email
    account_database["users"][str(user_id)]["email"] = new_email
    return True
