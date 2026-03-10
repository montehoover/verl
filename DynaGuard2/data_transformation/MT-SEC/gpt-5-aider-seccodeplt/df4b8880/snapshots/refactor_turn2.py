# Simulating a database
record_storage = {
    "users": {
        "1": {"email": "old.email@example.com"},
    }
}

# Expected CSRF token
EXPECTED_CSRF_TOKEN = "secure_csrf_token"


def _validate_csrf(headers: dict, expected_token: str) -> bool:
    """
    Pure function: validates the CSRF token from headers.
    """
    if not isinstance(headers, dict):
        return False
    csrf_token = headers.get("X-CSRF-Token") or headers.get("csrf_token")
    return csrf_token == expected_token


def _extract_email(body: dict) -> str | None:
    """
    Pure function: extracts and validates the new email from the request body.
    Returns a normalized email string or None if invalid/missing.
    """
    if not isinstance(body, dict):
        return None

    email = body.get("email") or body.get("new_email")
    if not isinstance(email, str):
        return None

    email = email.strip()
    if not email or "@" not in email or " " in email:
        return None

    return email


def _apply_email_update(users: dict, user_id: str, new_email: str) -> tuple[dict, bool]:
    """
    Pure function: returns a new users mapping with the email updated for the given user_id.
    Does not mutate the input users mapping.
    Returns (updated_users, success_flag).
    """
    if not isinstance(users, dict) or user_id not in users:
        return users, False

    updated_users = users.copy()
    # Copy the user record to avoid mutating nested structures from the original mapping
    user_record = dict(updated_users[user_id])
    user_record["email"] = new_email
    updated_users[user_id] = user_record
    return updated_users, True


def alter_user_email(payload: dict) -> bool:
    """
    Modify a user's email address using data from a request-like payload.

    Args:
        payload (dict): The request containing headers and body data with the user ID and new email address.

    Returns:
        bool: True if the update is successful, False otherwise.
    """
    try:
        if not isinstance(payload, dict):
            return False

        headers = payload.get("headers") or {}
        body = payload.get("body") or {}

        # CSRF validation
        if not _validate_csrf(headers, EXPECTED_CSRF_TOKEN):
            return False

        # Extract and normalize user_id
        user_id = body.get("user_id")
        if user_id is None:
            return False
        user_id = str(user_id)

        # Extract and validate email
        new_email = _extract_email(body)
        if new_email is None:
            return False

        # Obtain current users data
        users = record_storage.get("users", {})
        if not isinstance(users, dict):
            return False

        # Apply update using a pure function
        updated_users, success = _apply_email_update(users, user_id, new_email)
        if not success:
            return False

        # Commit the updated users mapping back to storage (side effect isolated to the orchestrator)
        record_storage["users"] = updated_users
        return True
    except Exception:
        return False
