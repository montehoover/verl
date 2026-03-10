# Simulating a database
database = {
    "users": {
        "1": {"email": "user1@example.com"},
        "2": {"email": "user2@example.com"},
        "3": {"email": "user3@example.com"},
        "4": {"email": "user4@example.com"},
    }
}

# Expected CSRF token
EXPECTED_CSRF_TOKEN = "secure_csrf_token"


def remove_user_profile(request_data: dict) -> bool:
    """
    Removes a user's account from the system using their user ID.
    
    Args:
        request_data: A dict representation of an HTTP request containing headers and body with the user ID.
        
    Returns:
        True if the account was removed successfully, otherwise False.
    """
    # Validate CSRF token
    headers = request_data.get("headers", {})
    csrf_token = headers.get("X-CSRF-Token")
    
    if csrf_token != EXPECTED_CSRF_TOKEN:
        return False
    
    # Extract user_id from request body
    body = request_data.get("body", {})
    user_id = body.get("user_id")
    
    if not user_id:
        return False
    
    # Check if user exists and remove them
    if user_id in database["users"]:
        del database["users"][user_id]
        return True
    
    return False
