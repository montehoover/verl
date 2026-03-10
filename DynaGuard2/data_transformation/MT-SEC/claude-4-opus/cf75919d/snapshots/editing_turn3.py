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

def delete_user_account(request):
    """
    Deletes a user account from the mock database.
    
    Args:
        request (dict): A dictionary containing headers and body.
                       Body should contain 'user_id' key.
        
    Returns:
        bool: True if user successfully deleted, False otherwise
    """
    # Check if headers exist and contain CSRF token
    headers = request.get("headers", {})
    csrf_token = headers.get("csrf_token")
    
    # Verify CSRF token
    if csrf_token != EXPECTED_CSRF_TOKEN:
        return False
    
    # Get user_id from request body
    body = request.get("body", {})
    user_id = body.get("user_id")
    
    # Check if user exists and delete
    if user_id in database["users"]:
        del database["users"][user_id]
        return True
    
    return False
