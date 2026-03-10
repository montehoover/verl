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
    # Extract CSRF token from headers
    csrf_token = request.get("headers", {}).get("X-CSRF-Token")
    
    # Verify CSRF token
    if csrf_token != EXPECTED_CSRF_TOKEN:
        return False
    
    # Extract user ID from request body
    user_id = request.get("body", {}).get("user_id")
    
    # Check if user exists in database
    if user_id and user_id in database["users"]:
        # Delete the user
        del database["users"][user_id]
        return True
    
    return False
