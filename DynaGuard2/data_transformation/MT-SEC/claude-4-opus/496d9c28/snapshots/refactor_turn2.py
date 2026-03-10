# Simulating a database
user_database = {
    "users": {
        "1": {"email": "old.email@example.com"},
    }
}

# Expected CSRF token
VALID_CSRF_TOKEN = "secure_csrf_token"

def validate_csrf_token(headers: dict, expected_token: str) -> bool:
    """Validate CSRF token from request headers."""
    csrf_token = headers.get("X-CSRF-Token")
    return csrf_token == expected_token

def update_user_email(user_id: str, new_email: str, database: dict) -> bool:
    """Update user's email in the database."""
    if user_id not in database["users"]:
        return False
    
    database["users"][user_id]["email"] = new_email
    return True

def change_user_email(new_request: dict) -> bool:
    # Check if request has required structure
    if "headers" not in new_request or "body" not in new_request:
        return False
    
    # Validate CSRF token
    if not validate_csrf_token(new_request["headers"], VALID_CSRF_TOKEN):
        return False
    
    # Get user ID and new email from body
    body = new_request.get("body", {})
    user_id = body.get("user_id")
    new_email = body.get("new_email")
    
    # Validate inputs
    if not user_id or not new_email:
        return False
    
    # Update email in database
    return update_user_email(user_id, new_email, user_database)
