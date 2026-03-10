# Simulating a database
record_storage = {
    "users": {
        "1": {"email": "old.email@example.com"},
    }
}

# Expected CSRF token
EXPECTED_CSRF_TOKEN = "secure_csrf_token"


def validate_csrf_token(headers: dict, expected_token: str) -> bool:
    """Validate CSRF token from request headers."""
    csrf_token = headers.get("X-CSRF-Token", "")
    return csrf_token == expected_token


def extract_user_data(body: dict) -> tuple[str, str]:
    """Extract user ID and email from request body."""
    user_id = body.get("user_id", "")
    new_email = body.get("email", "")
    return user_id, new_email


def user_exists(user_id: str, database: dict) -> bool:
    """Check if user exists in the database."""
    return user_id in database.get("users", {})


def update_user_email(user_id: str, new_email: str, database: dict) -> bool:
    """Update user's email in the database."""
    if user_id in database.get("users", {}):
        database["users"][user_id]["email"] = new_email
        return True
    return False


def alter_user_email(payload: dict) -> bool:
    """
    Modify a user's email address in the web application.
    
    Args:
        payload: dict, The request containing headers and body data with the user ID and new email address.
    
    Returns:
        bool, True if the update is successful, False if it fails.
    """
    try:
        # Validate CSRF token
        headers = payload.get("headers", {})
        if not validate_csrf_token(headers, EXPECTED_CSRF_TOKEN):
            return False
        
        # Extract user data
        body = payload.get("body", {})
        user_id, new_email = extract_user_data(body)
        
        # Check if user exists
        if not user_exists(user_id, record_storage):
            return False
        
        # Update the email
        return update_user_email(user_id, new_email, record_storage)
        
    except Exception:
        return False
