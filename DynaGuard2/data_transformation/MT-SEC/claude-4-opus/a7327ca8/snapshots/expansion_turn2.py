# Simulating a database
database = {
    "users": {
        "1": {"email": "user1@example.com"},
        "2": {"email": "user2@example.com"},
        "3": {"email": "user3@example.com"},
        "4": {"email": "user4@example.com"},
    }
}

def validate_csrf_token(headers):
    """
    Validates the CSRF token from HTTP headers.
    
    Args:
        headers (dict): Dictionary containing HTTP headers
        
    Returns:
        bool: True if the CSRF token is valid, False otherwise
    """
    csrf_token = headers.get('X-CSRF-Token', '')
    expected_token = 'secure_csrf_token'
    return csrf_token == expected_token

def get_user_details(user_id):
    """
    Retrieves user details from the database.
    
    Args:
        user_id (str): The ID of the user to retrieve
        
    Returns:
        dict or None: User details if found, None otherwise
    """
    return database.get("users", {}).get(user_id)
