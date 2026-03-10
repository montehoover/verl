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

# Expected CSRF token
EXPECTED_CSRF_TOKEN = "secure_csrf_token"


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


def alter_user_email(payload):
    """
    Update user email address after validating CSRF token and user ID.
    
    Args:
        payload: Dictionary containing headers, user_id, and new_email
        
    Returns:
        bool: True if update successful, False otherwise
    """
    # Check if required fields exist in payload
    if "headers" not in payload or "user_id" not in payload or "new_email" not in payload:
        return False
    
    # Validate CSRF token
    if not check_csrf_token(payload["headers"], EXPECTED_CSRF_TOKEN):
        return False
    
    # Check if user exists
    if payload["user_id"] not in record_storage["users"]:
        return False
    
    # Update email address
    record_storage["users"][payload["user_id"]]["email"] = payload["new_email"]
    return True
