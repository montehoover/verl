# Simulating a database
database = {
    "users": {
        "1": {"phone": "123-456-7890"},
    }
}

# Expected CSRF token
EXPECTED_CSRF_TOKEN = "secure_csrf_token"

def alter_user_phone(service_request: dict) -> bool:
    # Extract CSRF token from headers
    headers = service_request.get("headers", {})
    csrf_token = headers.get("X-CSRF-Token")
    
    # Validate CSRF token
    if csrf_token != EXPECTED_CSRF_TOKEN:
        return False
    
    # Extract body data
    body = service_request.get("body", {})
    user_id = body.get("user_id")
    new_phone = body.get("phone")
    
    # Validate required fields
    if not user_id or not new_phone:
        return False
    
    # Check if user exists in database
    if user_id not in database.get("users", {}):
        return False
    
    # Update phone number
    database["users"][user_id]["phone"] = new_phone
    return True
