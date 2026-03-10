# Simulating a database
database = {
    "users": {
        "1": {"phone": "123-456-7890"},
    }
}

# Expected CSRF token
EXPECTED_CSRF_TOKEN = "secure_csrf_token"

def modify_user_phone(user_request: dict) -> bool:
    # Check if request has required structure
    if "headers" not in user_request or "body" not in user_request:
        return False
    
    # Validate CSRF token
    headers = user_request.get("headers", {})
    csrf_token = headers.get("csrf_token", "")
    if csrf_token != EXPECTED_CSRF_TOKEN:
        return False
    
    # Extract user ID and new phone number from body
    body = user_request.get("body", {})
    user_id = body.get("user_id", "")
    new_phone = body.get("phone", "")
    
    # Check if user exists in database
    if user_id not in database.get("users", {}):
        return False
    
    # Update phone number
    database["users"][user_id]["phone"] = new_phone
    return True
