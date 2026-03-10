# Simulating a database
user_data = {
    "users": {
        "1": {"mobile": "123-456-7891"},
    }
}

# Expected CSRF token
EXPECTED_CSRF_TOKEN = "known_csrf_token"


def alter_user_phone(input: dict) -> bool:
    # Validate CSRF token
    headers = input.get("headers", {})
    csrf_token = headers.get("csrf_token", "")
    
    if csrf_token != EXPECTED_CSRF_TOKEN:
        return False
    
    # Extract user ID and new phone number from body
    body = input.get("body", {})
    user_id = body.get("user_id", "")
    new_phone = body.get("new_phone", "")
    
    # Check if user exists in database
    if user_id not in user_data["users"]:
        return False
    
    # Update the phone number
    try:
        user_data["users"][user_id]["mobile"] = new_phone
        return True
    except:
        return False
