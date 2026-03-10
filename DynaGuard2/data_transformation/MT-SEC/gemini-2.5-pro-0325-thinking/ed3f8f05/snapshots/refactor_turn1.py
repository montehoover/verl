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
        request_data: A dictionary representing the HTTP request,
                      including headers and body with user data.
                      Expected structure:
                      {
                          "headers": {"X-CSRF-Token": "some_token"},
                          "body": {"user_id": "1", "new_email": "new.email@example.com"}
                      }

    Returns:
        True if the email was successfully updated, False otherwise.
    """
    if not isinstance(request_data, dict):
        return False

    headers = request_data.get("headers")
    body = request_data.get("body")

    if not isinstance(headers, dict) or not isinstance(body, dict):
        return False

    csrf_token = headers.get("X-CSRF-Token")
    if csrf_token != EXPECTED_CSRF_TOKEN:
        return False

    user_id = body.get("user_id")
    new_email = body.get("new_email")

    if not user_id or not new_email:
        return False

    if user_id in database["users"]:
        database["users"][user_id]["email"] = new_email
        return True
    
    return False
