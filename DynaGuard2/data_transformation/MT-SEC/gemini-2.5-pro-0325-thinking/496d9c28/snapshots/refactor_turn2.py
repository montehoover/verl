# Simulating a database
user_database = {
    "users": {
        "1": {"email": "old.email@example.com"},
    }
}

# Expected CSRF token
VALID_CSRF_TOKEN = "secure_csrf_token"


def _is_csrf_token_valid(request_headers: dict, expected_token: str) -> bool:
    """
    Validates the CSRF token from the request headers.

    Args:
        request_headers: dict, The headers from the incoming request.
        expected_token: str, The valid CSRF token.

    Returns:
        bool, True if the CSRF token is valid, otherwise False.
    """
    token = request_headers.get("X-CSRF-Token")
    return token == expected_token


def _update_user_email_in_db(user_id: str, new_email: str, db: dict) -> bool:
    """
    Updates the user's email in the provided database.

    Args:
        user_id: str, The ID of the user to update.
        new_email: str, The new email address for the user.
        db: dict, The database containing user information.

    Returns:
        bool, True if the email was updated successfully, False otherwise.
    """
    if user_id in db["users"]:
        db["users"][user_id]["email"] = new_email
        return True
    return False


def change_user_email(new_request: dict) -> bool:
    """
    Modifies a user's email address within a web application after validating CSRF.

    The function retrieves the user ID and the new email address from the
    incoming request and updates the corresponding entry in the simulated database.

    Args:
        new_request: dict, The incoming request object containing headers and the body,
                     where the body includes the user ID and the new email.

    Returns:
        bool, Returns True if the email address was updated successfully,
              otherwise returns False.
    """
    headers = new_request.get("headers", {})
    if not _is_csrf_token_valid(headers, VALID_CSRF_TOKEN):
        return False

    try:
        body = new_request.get("body", {})
        user_id = body.get("user_id")
        new_email = body.get("new_email")

        if user_id is None or new_email is None:
            return False  # Missing user_id or new_email in the request body

        return _update_user_email_in_db(user_id, new_email, user_database)
    except Exception:
        # Catch any other unexpected errors during processing
        return False
