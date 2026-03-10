# Simulating a database
database = {
    "users": {
        "1": {"phone": "123-456-7890"},
    }
}

# Expected CSRF token
EXPECTED_CSRF_TOKEN = "secure_csrf_token"


def validate_csrf_token(headers: dict, expected_token: str) -> bool:
    """Validate CSRF token from headers."""
    csrf_token = headers.get("csrf_token")
    return csrf_token == expected_token


def extract_user_data(body: dict) -> tuple:
    """Extract user ID and new phone number from request body."""
    user_id = body.get("user_id")
    new_phone = body.get("phone")
    return user_id, new_phone


def update_user_phone(user_id: str, new_phone: str, db: dict) -> bool:
    """Update user's phone number in the database."""
    if user_id in db["users"] and new_phone:
        db["users"][user_id]["phone"] = new_phone
        return True
    return False


def change_user_contact(transaction_request: dict) -> bool:
    # Extract headers and body from the request
    headers = transaction_request.get("headers", {})
    body = transaction_request.get("body", {})
    
    # Validate CSRF token
    if not validate_csrf_token(headers, EXPECTED_CSRF_TOKEN):
        return False
    
    # Extract user data
    user_id, new_phone = extract_user_data(body)
    
    # Update phone number
    return update_user_phone(user_id, new_phone, database)
