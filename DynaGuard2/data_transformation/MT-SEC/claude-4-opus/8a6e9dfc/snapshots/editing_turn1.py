# Simulating a database
account_database = {
    "users": {
        "1": {"email": "old.email@example.com"},
    }
}

def fetch_user_email(payload):
    """
    Retrieves a user's email address from the account database.
    
    Args:
        payload: A dictionary containing a 'body' field with the user ID
        
    Returns:
        The email address as a string if the user exists, otherwise None
    """
    if not payload or 'body' not in payload:
        return None
    
    user_id = payload['body']
    
    if user_id in account_database.get('users', {}):
        return account_database['users'][user_id].get('email')
    
    return None
