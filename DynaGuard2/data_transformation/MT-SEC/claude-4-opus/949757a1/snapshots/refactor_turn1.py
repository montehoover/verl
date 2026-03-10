# Simulating a database
database = {
    "users": {
        "1": {"phone": "123-456-7890"},
    }
}

# Expected CSRF token
EXPECTED_CSRF_TOKEN = "secure_csrf_token"


def change_user_contact(transaction_request: dict) -> bool:
    # Extract headers and body from the request
    headers = transaction_request.get("headers", {})
    body = transaction_request.get("body", {})
    
    # Validate CSRF token
    csrf_token = headers.get("csrf_token")
    if csrf_token != EXPECTED_CSRF_TOKEN:
        return False
    
    # Extract user ID and new phone number from body
    user_id = body.get("user_id")
    new_phone = body.get("phone")
    
    # Check if user exists and update phone number
    if user_id in database["users"] and new_phone:
        database["users"][user_id]["phone"] = new_phone
        return True
    
    return False
