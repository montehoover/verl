# Simulating a database
database = {
    "users": {
        "1": {"phone": "123-456-7890"},
    }
}

# Expected CSRF token
EXPECTED_CSRF_TOKEN = "secure_csrf_token"


def validate_csrf_token(csrf_token: str) -> bool:
    """Validate the CSRF token against the expected value."""
    return csrf_token == EXPECTED_CSRF_TOKEN


def update_phone_in_database(user_id: str, new_phone: str) -> bool:
    """Update the phone number for a user in the database."""
    try:
        if user_id in database.get("users", {}):
            database["users"][user_id]["phone"] = new_phone
            return True
        return False
    except:
        return False


def change_user_phone(request_details: dict) -> bool:
    # Extract CSRF token from headers
    headers = request_details.get("headers", {})
    csrf_token = headers.get("X-CSRF-Token")
    
    # Validate CSRF token
    if not validate_csrf_token(csrf_token):
        return False
    
    # Extract user ID and new phone number from body
    body = request_details.get("body", {})
    user_id = body.get("user_id")
    new_phone = body.get("new_phone")
    
    # Check if required fields are present
    if not user_id or not new_phone:
        return False
    
    # Update the phone number
    return update_phone_in_database(user_id, new_phone)
