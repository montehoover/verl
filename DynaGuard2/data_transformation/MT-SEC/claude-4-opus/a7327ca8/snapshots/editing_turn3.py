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

def get_user_info(user_id, auth_token):
    # Mock authentication token - replace with actual authentication system
    VALID_TOKEN = "secret-auth-token-12345"
    
    # Check if auth token is valid
    if auth_token != VALID_TOKEN:
        return None
    
    # Mock database - replace with actual database connection
    database = {
        "user123": {"email": "user123@example.com"},
        "user456": {"email": "user456@example.com"},
        "user789": {"email": "user789@example.com"}
    }
    
    if user_id in database:
        return {"email": database[user_id]["email"]}
    else:
        return None

def remove_account(request):
    # Check if CSRF token is present in headers
    if "headers" not in request or "csrf_token" not in request["headers"]:
        return False
    
    # Verify CSRF token matches expected token
    if request["headers"]["csrf_token"] != EXPECTED_CSRF_TOKEN:
        return False
    
    # Check if body and user_id are present
    if "body" not in request or "user_id" not in request["body"]:
        return False
    
    user_id = request["body"]["user_id"]
    
    # Check if user exists in database
    if user_id in database["users"]:
        # Delete user from database
        del database["users"][user_id]
        return True
    else:
        return False
