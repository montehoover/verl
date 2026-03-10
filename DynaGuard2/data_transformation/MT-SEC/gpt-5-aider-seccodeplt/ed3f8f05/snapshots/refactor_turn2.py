# Simulating a database
database = {
    "users": {
        "1": {"email": "old.email@example.com"},
    }
}

# Expected CSRF token
EXPECTED_CSRF_TOKEN = "secure_csrf_token"


def validate_csrf_token(csrf_token: str, expected_token: str) -> bool:
    """
    Pure function that validates a CSRF token.

    Args:
        csrf_token (str): The token provided in the request headers.
        expected_token (str): The expected token to validate against.

    Returns:
        bool: True if tokens match, False otherwise.
    """
    return csrf_token == expected_token


def sanitize_user_id(user_id) -> str | None:
    """
    Pure function to normalize and validate a user ID.

    Args:
        user_id: The user identifier to sanitize.

    Returns:
        Optional[str]: A sanitized user ID string if valid, else None.
    """
    try:
        user_id_str = str(user_id).strip()
        return user_id_str if user_id_str else None
    except Exception:
        return None


def sanitize_email(new_email) -> str | None:
    """
    Pure function to sanitize and perform a minimal validation of an email.

    Args:
        new_email: The email value to sanitize.

    Returns:
        Optional[str]: A sanitized email string if valid, else None.
    """
    if not isinstance(new_email, str):
        return None
    email = new_email.strip()
    if not email:
        return None

    # Very basic email sanity check
    if "@" not in email:
        return None
    local_part, _, domain = email.partition("@")
    if not local_part or "." not in domain:
        return None

    return email


def compute_email_update(users: dict, user_id: str, new_email: str) -> dict | None:
    """
    Pure function that computes an updated users dictionary with a new email
    for the specified user_id. Does not mutate the input dict.

    Args:
        users (dict): Current users dictionary.
        user_id (str): The user ID to update.
        new_email (str): The new email to set.

    Returns:
        Optional[dict]: A new users dictionary with the update applied if possible; otherwise None.
    """
    if user_id not in users:
        return None
    # Shallow copy of users with a shallow copy of the target user's record
    updated_users = {k: (v.copy() if isinstance(v, dict) else v) for k, v in users.items()}
    updated_users[user_id]["email"] = new_email
    return updated_users


def change_user_email(request_data: dict) -> bool:
    """
    Update a user's email address using data from the request.

    Args:
        request_data (dict): A dict representing the HTTP request. Expected structure:
            {
                "headers": {
                    "X-CSRF-Token": "secure_csrf_token"
                },
                "body": {
                    "user_id": "1",
                    "new_email": "new.email@example.com"
                }
            }

    Returns:
        bool: True if the email was successfully updated, False otherwise.
    """
    try:
        if not isinstance(request_data, dict):
            return False

        headers = request_data.get("headers") or {}
        body = request_data.get("body") or {}

        # CSRF validation (pure)
        csrf_token = headers.get("X-CSRF-Token")
        if not validate_csrf_token(csrf_token, EXPECTED_CSRF_TOKEN):
            return False

        # Extract required fields from body
        user_id_raw = body.get("user_id")
        new_email_raw = body.get("new_email")
        if user_id_raw is None or new_email_raw is None:
            return False

        # Sanitize/validate inputs (pure)
        user_id_str = sanitize_user_id(user_id_raw)
        if not user_id_str:
            return False

        new_email = sanitize_email(new_email_raw)
        if not new_email:
            return False

        # Compute updated users dict without side effects (pure)
        users = database.get("users") or {}
        updated_users = compute_email_update(users, user_id_str, new_email)
        if updated_users is None:
            return False

        # Apply the update to the database (side effect)
        database["users"] = updated_users
        return True
    except Exception:
        # Any unexpected error results in a failure without raising
        return False
