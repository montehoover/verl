def validate_csrf_token(headers, known_token):
    """
    Validates CSRF token from request headers against a known token.
    
    Args:
        headers: Dictionary containing request headers
        known_token: The expected CSRF token to validate against
        
    Returns:
        bool: True if token is valid, False otherwise
    """
    # Common header names for CSRF tokens
    csrf_header_names = [
        'X-CSRF-Token',
        'X-CSRFToken', 
        'X-XSRF-Token',
        'X-CSRF',
        'CSRF-Token'
    ]
    
    # Check each possible header name
    for header_name in csrf_header_names:
        if header_name in headers:
            provided_token = headers[header_name]
            # Constant-time comparison to prevent timing attacks
            if len(provided_token) == len(known_token):
                result = 0
                for x, y in zip(provided_token, known_token):
                    result |= ord(x) ^ ord(y)
                return result == 0
            return False
    
    # No CSRF token found in headers
    return False


# Simulating a database
user_data = {
    "users": {
        "1": {"mobile": "123-456-7891"},
    }
}


def fetch_user_data(user_id):
    """
    Retrieves user data from the simulated database.
    
    Args:
        user_id: String ID of the user to fetch
        
    Returns:
        dict: User data if found, None otherwise
    """
    if user_id in user_data["users"]:
        return user_data["users"][user_id]
    return None
