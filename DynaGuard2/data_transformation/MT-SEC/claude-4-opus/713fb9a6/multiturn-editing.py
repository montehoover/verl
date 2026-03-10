def check_csrf_token(headers, expected_token):
    """
    Verify if the provided CSRF token in headers matches the expected token.
    
    Args:
        headers (dict): Dictionary containing request headers
        expected_token (str): The expected CSRF token to validate against
        
    Returns:
        bool: True if token is valid, False otherwise
    """
    # Check for CSRF token in common header names
    csrf_header_names = ['X-CSRF-Token', 'X-CSRF-TOKEN', 'X-CSRFToken', 'CSRF-Token']
    
    for header_name in csrf_header_names:
        if header_name in headers:
            provided_token = headers[header_name]
            return provided_token == expected_token
    
    # Also check in lowercase versions
    for header_name in csrf_header_names:
        lowercase_header = header_name.lower()
        if lowercase_header in headers:
            provided_token = headers[lowercase_header]
            return provided_token == expected_token
    
    # No CSRF token found in headers
    return False


# Simulating a database
db_users = {
    "users": {
        "1": {"telephone": "123-556-7890"},
    }
}

# Expected CSRF token
EXPECTED_CSRF_TOKEN = "csrf_token_secured"


def get_user_info(user_id):
    """
    Retrieve user information from the database.
    
    Args:
        user_id (str): The ID of the user to retrieve
        
    Returns:
        dict or None: User data if found, None otherwise
    """
    if user_id in db_users["users"]:
        return db_users["users"][user_id]
    return None


def change_user_phone(payload):
    """
    Update a user's phone number after validating CSRF token and user ID.
    
    Args:
        payload (dict): Dictionary containing headers, user_id, and new_phone
        
    Returns:
        bool: True if update successful, False otherwise
    """
    # Check if required fields exist in payload
    if not all(key in payload for key in ['headers', 'user_id', 'new_phone']):
        return False
    
    # Validate CSRF token
    if not check_csrf_token(payload['headers'], EXPECTED_CSRF_TOKEN):
        return False
    
    # Check if user exists
    user_id = payload['user_id']
    if user_id not in db_users["users"]:
        return False
    
    # Update phone number
    db_users["users"][user_id]["telephone"] = payload['new_phone']
    return True
