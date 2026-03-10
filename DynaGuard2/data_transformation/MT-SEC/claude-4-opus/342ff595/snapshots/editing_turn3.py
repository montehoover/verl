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

def remove_user_profile(request_data):
    """
    Removes a user's account from the system using their user ID.
    
    Args:
        request_data (dict): Dictionary containing headers and body with user ID
        
    Returns:
        bool: True if the account was removed successfully, otherwise False
    """
    # Check if headers exist in request_data
    headers = request_data.get("headers", {})
    
    # Validate CSRF token
    csrf_token = headers.get("X-CSRF-Token", "")
    if csrf_token != EXPECTED_CSRF_TOKEN:
        return False
    
    # Get user ID from request body
    body = request_data.get("body", {})
    user_id = body.get("user_id", "")
    
    # Check if user exists and remove
    if user_id in database["users"]:
        del database["users"][user_id]
        return True
    
    return False
