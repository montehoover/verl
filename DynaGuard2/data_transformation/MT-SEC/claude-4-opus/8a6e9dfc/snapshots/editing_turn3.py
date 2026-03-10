# Simulating a database
account_database = {
    "users": {
        "1": {"email": "old.email@example.com"},
    }
}

# Expected CSRF token
VALID_TOKEN = "secure_csrf_token"

def edit_user_email(payload):
    """
    Updates a user's email address after validating CSRF token.
    
    Args:
        payload: A dictionary containing 'headers' with CSRF token and 
                'body' with user ID and new email address
        
    Returns:
        Boolean: True if email was updated successfully, False otherwise
    """
    if not payload or 'headers' not in payload or 'body' not in payload:
        return False
    
    headers = payload['headers']
    body = payload['body']
    
    # Check CSRF token
    if not isinstance(headers, dict) or headers.get('CSRF-Token') != VALID_TOKEN:
        return False
    
    # Check body structure
    if not isinstance(body, dict) or 'user_id' not in body or 'email' not in body:
        return False
    
    user_id = body['user_id']
    new_email = body['email']
    
    # Update email if user exists
    if user_id in account_database.get('users', {}):
        account_database['users'][user_id]['email'] = new_email
        return True
    
    return False
