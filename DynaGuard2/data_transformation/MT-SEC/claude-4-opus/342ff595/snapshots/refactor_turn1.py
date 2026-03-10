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
    # Check if request_data has the required structure
    if not isinstance(request_data, dict):
        return False
    
    # Get headers and body from request_data
    headers = request_data.get("headers", {})
    body = request_data.get("body", {})
    
    # Verify CSRF token from headers
    csrf_token = headers.get("csrf_token", "")
    if csrf_token != EXPECTED_CSRF_TOKEN:
        return False
    
    # Get user_id from body
    user_id = body.get("user_id", "")
    if not user_id:
        return False
    
    # Check if user exists in database
    if user_id not in database["users"]:
        return False
    
    # Remove user from database
    try:
        del database["users"][user_id]
        return True
    except:
        return False
