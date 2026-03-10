# Simulating a database
database = {
    "users": {
        "1": {"phone": "123-456-7890"},
    }
}

# Expected CSRF token
EXPECTED_CSRF_TOKEN = "secure_csrf_token"


def change_user_phone(request_details: dict) -> bool:
    # Extract CSRF token from headers
    headers = request_details.get("headers", {})
    csrf_token = headers.get("X-CSRF-Token")
    
    # Validate CSRF token
    if csrf_token != EXPECTED_CSRF_TOKEN:
        return False
    
    # Extract user ID and new phone number from body
    body = request_details.get("body", {})
    user_id = body.get("user_id")
    new_phone = body.get("new_phone")
    
    # Check if user exists in database
    if user_id not in database.get("users", {}):
        return False
    
    # Check if required fields are present
    if not user_id or not new_phone:
        return False
    
    # Update the phone number
    try:
        database["users"][user_id]["phone"] = new_phone
        return True
    except:
        return False
