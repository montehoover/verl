# Simulating a database
db_store = {
    "users": {
        "1": {"cell": "123-456-7890"},
    }
}

# Expected CSRF token
EXPECTED_CSRF_TOKEN = "secure_csrf_token_value"


def validate_csrf_token(headers: dict, expected_token: str) -> bool:
    """Validate CSRF token from request headers."""
    csrf_token = headers.get("csrf_token", "")
    return csrf_token == expected_token


def extract_phone_update_data(body: dict) -> tuple:
    """Extract user ID and new phone number from request body."""
    user_id = body.get("user_id", "")
    new_phone = body.get("new_phone", "")
    return user_id, new_phone


def update_user_phone_in_db(user_id: str, new_phone: str, database: dict) -> bool:
    """Update user's phone number in the database."""
    if user_id in database["users"]:
        database["users"][user_id]["cell"] = new_phone
        return True
    return False


def modify_user_phone(request_data: dict) -> bool:
    # Validate request structure
    if "headers" not in request_data or "body" not in request_data:
        return False
    
    # Validate CSRF token
    if not validate_csrf_token(request_data["headers"], EXPECTED_CSRF_TOKEN):
        return False
    
    # Extract update data
    user_id, new_phone = extract_phone_update_data(request_data["body"])
    
    # Validate extracted data
    if not user_id or not new_phone:
        return False
    
    # Update database
    return update_user_phone_in_db(user_id, new_phone, db_store)
