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


def terminate_user_account(request_data: dict) -> bool:
    # Check if headers exist and contain CSRF token
    if "headers" not in request_data:
        return False
    
    headers = request_data["headers"]
    if "csrf_token" not in headers or headers["csrf_token"] != EXPECTED_CSRF_TOKEN:
        return False
    
    # Check if body exists and contains user_id
    if "body" not in request_data:
        return False
    
    body = request_data["body"]
    if "user_id" not in body:
        return False
    
    user_id = body["user_id"]
    
    # Check if user exists in database and delete
    if user_id in database["users"]:
        del database["users"][user_id]
        return True
    
    return False
