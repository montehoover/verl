# Simulating a database
database = {
    "users": {
        "1": {"phone": "123-456-7890"},
    }
}

# Expected CSRF token
EXPECTED_CSRF_TOKEN = "secure_csrf_token"

def validate_csrf_token(headers: dict, expected_token: str) -> bool:
    """Validate CSRF token from request headers."""
    csrf_token = headers.get("csrf_token", "")
    return csrf_token == expected_token

def update_user_phone_in_db(database: dict, user_id: str, new_phone: str) -> bool:
    """Update user's phone number in the database."""
    if user_id not in database.get("users", {}):
        return False
    database["users"][user_id]["phone"] = new_phone
    return True

def extract_request_data(user_request: dict) -> tuple:
    """Extract headers and body from the request."""
    headers = user_request.get("headers", {})
    body = user_request.get("body", {})
    return headers, body

def modify_user_phone(user_request: dict) -> bool:
    # Check if request has required structure
    if "headers" not in user_request or "body" not in user_request:
        return False
    
    # Extract request data
    headers, body = extract_request_data(user_request)
    
    # Validate CSRF token
    if not validate_csrf_token(headers, EXPECTED_CSRF_TOKEN):
        return False
    
    # Extract user ID and new phone number from body
    user_id = body.get("user_id", "")
    new_phone = body.get("phone", "")
    
    # Update phone number in database
    return update_user_phone_in_db(database, user_id, new_phone)
