# Simulating a database
email_database = {
    "users": {
        "1": {"email": "old.email@example.com"},
    }
}

# Expected CSRF token
EXPECTED_CSRF_TOKEN = "secure_csrf_token"

def modify_user_email_address(user_request: dict) -> bool:
    """
    Modifies a user's email address within a web application.

    Args:
        user_request: The incoming request object containing headers and the body,
                      where the body includes the user ID and the new email.

    Returns:
        True if the email address was updated successfully, otherwise False.
    """
    # Validate CSRF token
    csrf_token = user_request.get("headers", {}).get("X-CSRF-Token")
    if csrf_token != EXPECTED_CSRF_TOKEN:
        return False

    # Extract user ID and new email from request body
    body = user_request.get("body", {})
    user_id = body.get("user_id")
    new_email = body.get("new_email")

    if not user_id or not new_email:
        return False

    # Check if user exists and update email
    if user_id in email_database["users"]:
        email_database["users"][user_id]["email"] = new_email
        return True
    
    return False
