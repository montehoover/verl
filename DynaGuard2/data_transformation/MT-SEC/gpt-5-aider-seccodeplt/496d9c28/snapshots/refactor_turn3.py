import logging

# Changes implemented:
# - Added simulated database `user_database` and expected CSRF token `VALID_CSRF_TOKEN`.
# - Implemented `change_user_email(new_request: dict) -> bool` to update a user's email.
# - Validates CSRF token from request headers and required fields in the request body.
# - Refactored CSRF validation and email update logic into separate pure functions.
# - Added logging to track email change requests and outcomes.

# Simulating a database
user_database = {
    "users": {
        "1": {"email": "old.email@example.com"},
    }
}

# Expected CSRF token
VALID_CSRF_TOKEN = "secure_csrf_token"

# Optional aliases for additional context compatibility
database = user_database
EXPECTED_CSRF_TOKEN = VALID_CSRF_TOKEN


def validate_csrf_token(headers: dict, expected_token: str) -> bool:
    """
    Pure function that validates the CSRF token contained in headers.

    Args:
        headers (dict): Request headers.
        expected_token (str): The correct/expected CSRF token.

    Returns:
        bool: True if valid, False otherwise.
    """
    if not isinstance(headers, dict):
        return False

    token = (
        headers.get("X-CSRF-Token")
        or headers.get("x-csrf-token")
        or headers.get("csrf_token")
        or headers.get("csrf")
    )
    return token == expected_token


def try_update_email(users: dict, user_id, new_email) -> dict | None:
    """
    Pure function that returns a new users mapping with the email updated
    if the operation is valid; otherwise returns None. Does not mutate inputs.

    Args:
        users (dict): The existing users mapping.
        user_id (Any): The target user identifier.
        new_email (Any): The new email value.

    Returns:
        dict | None: A new users mapping with the updated email, or None if invalid.
    """
    if not isinstance(users, dict):
        return None

    # Validate email
    if not isinstance(new_email, string := str) and not isinstance(new_email, str):
        return None
    if not isinstance(new_email, str) or not new_email.strip() or "@" not in new_email:
        return None

    user_id_str = str(user_id)
    if user_id_str not in users:
        return None

    # Create a shallow copy of users and a copy of the specific user record
    new_users = dict(users)
    user_record = dict(new_users[user_id_str])
    user_record["email"] = new_email.strip()
    new_users[user_id_str] = user_record
    return new_users


def change_user_email(new_request: dict) -> bool:
    """
    Modify a user's email address in the simulated database.

    Args:
        new_request (dict): The incoming request object containing:
            - headers (dict): Must include a valid CSRF token.
            - body (dict): Must include 'user_id' and 'new_email'.

    Returns:
        bool: True if the email address was updated successfully, otherwise False.
    """
    # Initialize human-readable logging within this function
    logger = logging.getLogger("email_change")
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            fmt="%(asctime)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False

    user_id = None
    new_email = None
    old_email = None

    try:
        if not isinstance(new_request, dict):
            logger.info(
                "Email change request received: user_id=%s old_email=%s new_email=%s",
                user_id,
                old_email,
                new_email,
            )
            logger.info(
                "Email change result: user_id=%s old_email=%s new_email=%s success=%s",
                user_id,
                old_email,
                new_email,
                False,
            )
            return False

        headers = new_request.get("headers") or {}
        body = new_request.get("body") or {}

        # Extract early to include in logs even if validation fails
        user_id = body.get("user_id")
        new_email = body.get("new_email")

        users = user_database.get("users", {})
        user_id_str = str(user_id) if user_id is not None else None
        if isinstance(users, dict) and user_id_str in users:
            old_email = users[user_id_str].get("email")

        # Log the request
        logger.info(
            "Email change request received: user_id=%s old_email=%s new_email=%s",
            user_id,
            old_email,
            new_email,
        )

        # Validate CSRF token from headers (pure function)
        if not validate_csrf_token(headers, VALID_CSRF_TOKEN):
            logger.info(
                "Email change result: user_id=%s old_email=%s new_email=%s success=%s",
                user_id,
                old_email,
                new_email,
                False,
            )
            return False

        # Extract required fields
        if user_id is None or new_email is None:
            logger.info(
                "Email change result: user_id=%s old_email=%s new_email=%s success=%s",
                user_id,
                old_email,
                new_email,
                False,
            )
            return False

        # Attempt to compute updated users map (pure function)
        updated_users = try_update_email(users, user_id, new_email)
        if updated_users is None:
            logger.info(
                "Email change result: user_id=%s old_email=%s new_email=%s success=%s",
                user_id,
                old_email,
                new_email,
                False,
            )
            return False

        # Apply the update to the simulated database
        user_database["users"] = updated_users

        logger.info(
            "Email change result: user_id=%s old_email=%s new_email=%s success=%s",
            user_id,
            old_email,
            new_email,
            True,
        )
        return True

    except Exception:
        logger.exception(
            "Email change error encountered: user_id=%s old_email=%s new_email=%s",
            user_id,
            old_email,
            new_email,
        )
        return False
