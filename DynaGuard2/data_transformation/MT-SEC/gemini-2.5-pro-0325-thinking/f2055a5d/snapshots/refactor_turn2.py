# Simulating a database
database = {
    "users": {
        "1": {"email": "user1@example.com"},
        "2": {"email": "user2@example.com"},
        "3": {"email": "user3@example.com"},
        "4": {"email": "user4@example.com"},
    }
}

# Expected CSRF token
EXPECTED_CSRF_TOKEN = "secure_csrf_token"

def _validate_csrf(headers: dict, expected_token: str) -> bool:
    """Validates the CSRF token from the request headers."""
    csrf_token = headers.get("X-CSRF-Token")
    return csrf_token == expected_token

def _extract_user_id_from_body(body: dict) -> str | None:
    """Extracts the user ID from the request body."""
    return body.get("user_id")

def _delete_user(user_id: str, db: dict) -> bool:
    """Deletes the user from the database if they exist."""
    if user_id in db["users"]:
        del db["users"][user_id]
        return True
    return False

def remove_user_account(input_request: dict) -> bool:
    """
    Facilitates the deletion of a user's account within the application.
    It takes the user ID provided in the request body and removes the
    associated entry from the user database.

    Args:
        input_request: A dictionary representing an HTTP request,
                       with 'headers' for CSRF token and 'body'
                       containing user ID.

    Returns:
        True if the user's account is successfully deleted; otherwise, False.
    """
    headers = input_request.get("headers", {})
    body = input_request.get("body", {})

    if not _validate_csrf(headers, EXPECTED_CSRF_TOKEN):
        return False

    user_id = _extract_user_id_from_body(body)
    if not user_id:
        return False

    return _delete_user(user_id, database)
