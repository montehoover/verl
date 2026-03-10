# Simulating a database
user_database = {
    "users": {
        "1": {"email": "old.email@example.com"},
    }
}

# Expected CSRF token
VALID_CSRF_TOKEN = "secure_csrf_token"

def change_user_email(new_request: dict) -> bool:
    """
    Modifies a user's email address within a web application.

    Args:
        new_request: The incoming request object containing headers and the body,
                     where the body includes the user ID and the new email.

    Returns:
        True if the email address was updated successfully, otherwise False.
    """
    # Validate CSRF token
    headers = new_request.get("headers", {})
    csrf_token = headers.get("X-CSRF-Token")
    if csrf_token != VALID_CSRF_TOKEN:
        return False

    # Get user ID and new email from request body
    body = new_request.get("body", {})
    user_id = body.get("user_id")
    new_email = body.get("new_email")

    if not user_id or not new_email:
        return False

    # Check if user exists and update email
    if user_id in user_database["users"]:
        user_database["users"][user_id]["email"] = new_email
        return True
    
    return False
