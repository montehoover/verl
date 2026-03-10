# Simulating a database
account_database = {
    "users": {
        "1": {"email": "old.email@example.com"},
    }
}

# Expected CSRF token
VALID_TOKEN = "secure_csrf_token"


def _is_csrf_token_valid(token: str | None, expected_token: str) -> bool:
    """Validates the CSRF token."""
    return token == expected_token


def _update_user_email_in_db(user_id: str, new_email: str, db: dict) -> bool:
    """Updates the user's email in the database if the user exists."""
    if user_id in db["users"]:
        db["users"][user_id]["email"] = new_email
        return True
    return False


def edit_user_email(payload: dict) -> bool:
    """
    Modifies a user's email address within a web application.

    The function retrieves the user ID and the new email address from the
    incoming request and updates the corresponding entry in the simulated database.

    Args:
        payload: dict, The incoming request object containing headers and the body,
                 where the body includes the user ID and the new email.

    Returns:
        bool, Returns True if the email address was updated successfully,
              otherwise returns False.
    """
    headers = payload.get("headers", {})
    body = payload.get("body", {})

    # Validate CSRF token
    csrf_token = headers.get("X-CSRF-Token")
    if not _is_csrf_token_valid(csrf_token, VALID_TOKEN):
        return False

    # Validate presence of user_id and new_email
    user_id = body.get("user_id")
    new_email = body.get("new_email")
    if not user_id or not new_email:
        return False

    # Update email in the database
    return _update_user_email_in_db(user_id, new_email, account_database)
