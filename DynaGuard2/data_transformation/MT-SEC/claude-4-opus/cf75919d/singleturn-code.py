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


def delete_user_account(request: dict) -> bool:
    """
    Deletes a user account from the database.
    
    Args:
        request: dict, The request object containing headers and body with user ID
        
    Returns:
        bool, True if the user is deleted successfully, False otherwise
    """
    # Check if request has required structure
    if not isinstance(request, dict):
        return False
    
    headers = request.get("headers", {})
    body = request.get("body", {})
    
    # Validate CSRF token
    csrf_token = headers.get("X-CSRF-Token")
    if csrf_token != EXPECTED_CSRF_TOKEN:
        return False
    
    # Get user ID from request body
    user_id = body.get("user_id")
    if not user_id:
        return False
    
    # Check if user exists and delete
    if user_id in database["users"]:
        del database["users"][user_id]
        return True
    
    return False
