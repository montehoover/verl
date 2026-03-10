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


def remove_account(request: dict) -> bool:
    """
    Remove a user account from the system.
    
    Args:
        request: dict, An object representing the HTTP request, containing headers and body with the user ID.
        
    Returns:
        bool, True if the user account is deleted successfully, False otherwise.
    """
    try:
        # Extract CSRF token from headers
        headers = request.get("headers", {})
        csrf_token = headers.get("X-CSRF-Token", "")
        
        # Verify CSRF token
        if csrf_token != EXPECTED_CSRF_TOKEN:
            return False
            
        # Extract user ID from request body
        body = request.get("body", {})
        user_id = body.get("user_id", "")
        
        # Check if user exists in database
        if user_id not in database["users"]:
            return False
            
        # Delete user from database
        del database["users"][user_id]
        return True
        
    except Exception:
        return False
