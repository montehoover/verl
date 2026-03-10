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


def _is_csrf_token_valid(headers: dict, expected_token: str) -> bool:
    """Validates the CSRF token from the request headers."""
    csrf_token = headers.get("X-CSRF-Token")
    return csrf_token == expected_token


def _delete_user_from_db(user_id: str, db: dict) -> bool:
    """Deletes a user from the database if the user exists."""
    if user_id in db["users"]:
        del db["users"][user_id]
        return True
    return False


def terminate_user_account(request_data: dict) -> bool:
    """
    Facilitates the deletion of a user's account within the application.

    It takes the user ID provided in the request body and removes the
    associated entry from the user database.

    Args:
        request_data: A dictionary representing an HTTP request,
                      with 'headers' for CSRF token and 'body'
                      containing user ID.

    Returns:
        True if the user's account is successfully deleted; otherwise, False.
    """
    headers = request_data.get("headers", {})
    if not _is_csrf_token_valid(headers, EXPECTED_CSRF_TOKEN):
        return False

    body = request_data.get("body", {})
    user_id = body.get("user_id")

    if not user_id:
        return False

    return _delete_user_from_db(user_id, database)
