# Simulating a database
record_storage = {
    "users": {
        "1": {"email": "old.email@example.com"},
    }
}

# Expected CSRF token
EXPECTED_CSRF_TOKEN = "secure_csrf_token"

def alter_user_email(payload: dict) -> bool:
    """
    Modifies a user’s email address in a web application.

    Args:
        payload: dict, The request containing headers and body data
                       with the user ID and new email address.
                       Expected structure:
                       {
                           "headers": {"X-CSRF-Token": "some_token"},
                           "body": {"user_id": "1", "new_email": "new.email@example.com"}
                       }

    Returns:
        bool, True if the update is successful, False if it fails.
    """
    if not payload:
        return False

    headers = payload.get("headers", {})
    body = payload.get("body", {})

    csrf_token = headers.get("X-CSRF-Token")
    user_id = body.get("user_id")
    new_email = body.get("new_email")

    if not all([csrf_token, user_id, new_email]):
        return False # Missing essential data

    if csrf_token != EXPECTED_CSRF_TOKEN:
        return False # CSRF token mismatch

    if user_id not in record_storage["users"]:
        return False # User not found

    record_storage["users"][user_id]["email"] = new_email
    return True
