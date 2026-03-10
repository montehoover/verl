# Simulating a database
account_database = {
    "users": {
        "1": {"email": "old.email@example.com"},
    }
}

# Expected CSRF token
VALID_TOKEN = "secure_csrf_token"


def validate_csrf_token(token: str) -> bool:
    """Validate the CSRF token against the expected value."""
    return token == VALID_TOKEN


def update_user_email_in_db(user_id: str, new_email: str) -> bool:
    """Update user's email in the database if user exists."""
    # Check if user exists before attempting update
    if user_id not in account_database["users"]:
        return False
    
    # Perform the email update
    account_database["users"][user_id]["email"] = new_email
    return True


def edit_user_email(payload: dict) -> bool:
    """
    Modify a user's email address within the web application.
    
    Args:
        payload: dict containing headers and body with user_id and new_email
        
    Returns:
        bool: True if email was updated successfully, False otherwise
    """
    # Validate payload structure
    if not payload or "headers" not in payload or "body" not in payload:
        return False
    
    # Extract and validate CSRF token for security
    csrf_token = payload["headers"].get("X-CSRF-Token", "")
    if not validate_csrf_token(csrf_token):
        return False
    
    # Extract user data from request body
    body = payload["body"]
    user_id = body.get("user_id")
    new_email = body.get("new_email")
    
    # Ensure all required fields are present
    if not user_id or not new_email:
        return False
    
    # Attempt to update the user's email in the database
    return update_user_email_in_db(user_id, new_email)
