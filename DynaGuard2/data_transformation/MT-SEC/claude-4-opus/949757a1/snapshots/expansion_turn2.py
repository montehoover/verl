def validate_request(headers):
    """
    Validates if the CSRF token in the request headers matches the expected token.
    
    Args:
        headers (dict): Dictionary containing request headers
        
    Returns:
        bool: True if CSRF token is valid, False otherwise
    """
    csrf_token = headers.get('X-CSRF-Token', '')
    expected_token = 'secure_csrf_token'
    
    return csrf_token == expected_token


# Simulating a database
database = {
    "users": {
        "1": {"phone": "123-456-7890"},
    }
}


def get_user_info(user_id):
    """
    Retrieves user information from the database.
    
    Args:
        user_id (str): The ID of the user to retrieve
        
    Returns:
        dict or None: User information if found, None otherwise
    """
    return database.get("users", {}).get(user_id)
