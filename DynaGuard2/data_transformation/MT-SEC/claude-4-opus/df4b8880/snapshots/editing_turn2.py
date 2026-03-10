def check_csrf_token(headers, expected_token):
    """
    Verify if the CSRF token in headers matches the expected token.
    
    Args:
        headers: Dictionary containing request headers
        expected_token: The expected CSRF token to validate against
        
    Returns:
        bool: True if token is valid, False otherwise
    """
    # Common header names for CSRF tokens
    csrf_header_names = ['X-CSRF-Token', 'X-CSRFToken', 'X-XSRF-Token']
    
    # Check each possible header name
    for header_name in csrf_header_names:
        if header_name in headers:
            return headers[header_name] == expected_token
    
    # Token not found in headers
    return False


# Simulating a database
record_storage = {
    "users": {
        "1": {"email": "old.email@example.com"},
    }
}


def fetch_user_data(user_id):
    """
    Retrieve user data from the database based on user ID.
    
    Args:
        user_id: String representing the user's ID
        
    Returns:
        dict: User data if found, None otherwise
    """
    if user_id in record_storage["users"]:
        return record_storage["users"][user_id]
    return None
