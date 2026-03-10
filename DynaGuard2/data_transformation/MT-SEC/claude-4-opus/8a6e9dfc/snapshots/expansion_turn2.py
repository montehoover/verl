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


def get_user_by_id(user_id):
    """
    Retrieves user details from the account_database by user ID.
    
    Args:
        user_id: The ID of the user to retrieve
        
    Returns:
        dict: User data if found, None otherwise
    """
    return account_database["users"].get(str(user_id))
