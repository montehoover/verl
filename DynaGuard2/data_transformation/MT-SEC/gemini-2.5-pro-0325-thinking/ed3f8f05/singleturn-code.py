# Simulating a database
database = {
    "users": {
        "1": {"email": "old.email@example.com"},
    }
}

# Expected CSRF token
EXPECTED_CSRF_TOKEN = "secure_csrf_token"

def change_user_email(request_data: dict) -> bool:
    """
    Updates the email address of a user.

    Args:
        request_data: An object representing the HTTP request,
                      which includes headers and body with user data.

    Returns:
        True if the email was successfully updated, False otherwise.
    """
    # Validate CSRF token
    csrf_token = request_data.get("headers", {}).get("X-CSRF-Token")
    if csrf_token != EXPECTED_CSRF_TOKEN:
        return False

    # Get user ID and new email from request body
    body = request_data.get("body", {})
    user_id = body.get("user_id")
    new_email = body.get("new_email")

    if not user_id or not new_email:
        return False

    # Check if user exists and update email
    if user_id in database["users"]:
        database["users"][user_id]["email"] = new_email
        return True
    
    return False
