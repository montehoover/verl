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


def remove_user_account(input_request: dict) -> bool:
    # Validate CSRF token
    headers = input_request.get("headers", {})
    csrf_token = headers.get("csrf_token", "")
    
    if csrf_token != EXPECTED_CSRF_TOKEN:
        return False
    
    # Extract user ID from request body
    body = input_request.get("body", {})
    user_id = body.get("user_id", "")
    
    # Check if user exists and delete
    if user_id in database["users"]:
        del database["users"][user_id]
        return True
    
    return False
