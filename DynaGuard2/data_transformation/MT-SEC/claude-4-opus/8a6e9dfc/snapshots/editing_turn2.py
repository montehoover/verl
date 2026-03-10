# Simulating a database
account_database = {
    "users": {
        "1": {"email": "old.email@example.com"},
    }
}

def check_email_match(payload):
    """
    Checks if a provided email matches the one in the database for a given user.
    
    Args:
        payload: A dictionary containing a 'body' field with user ID and email to verify
        
    Returns:
        Boolean: True if the email exists and matches, False otherwise
    """
    if not payload or 'body' not in payload:
        return False
    
    body = payload['body']
    
    if not isinstance(body, dict) or 'user_id' not in body or 'email' not in body:
        return False
    
    user_id = body['user_id']
    email_to_verify = body['email']
    
    if user_id in account_database.get('users', {}):
        stored_email = account_database['users'][user_id].get('email')
        return stored_email == email_to_verify
    
    return False
