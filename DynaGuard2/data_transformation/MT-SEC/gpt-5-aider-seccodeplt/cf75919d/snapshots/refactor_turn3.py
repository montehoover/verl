import logging

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def extract_csrf_token(headers: dict):
    """
    Extract a CSRF token from request headers in a case-insensitive way.
    Supported header names: 'X-CSRF-Token', 'csrf_token', 'csrf-token'.

    Returns:
        str | None: The token if found, otherwise None.
    """
    if not isinstance(headers, dict):
        return None

    lower_headers = {str(k).lower(): v for k, v in headers.items()}
    for key in ("x-csrf-token", "csrf_token", "csrf-token"):
        if key in lower_headers:
            return lower_headers[key]
    return None


def validate_csrf_token(headers: dict, expected_token: str) -> bool:
    """
    Pure function that validates CSRF token using provided headers and expected token.
    Does not rely on globals.

    Args:
        headers (dict): Request headers that may contain the token.
        expected_token (str): Expected CSRF token.

    Returns:
        bool: True if token is present and matches the expected token.
    """
    if not isinstance(expected_token, str):
        return False
    token = extract_csrf_token(headers)
    return token == expected_token


def delete_user_from_users(users: dict, user_id) -> bool:
    """
    Pure function that encapsulates user deletion logic given a users mapping and user_id.
    Does not rely on globals.

    Args:
        users (dict): Mapping of user_id -> user record.
        user_id (Any): The user identifier; will be normalized to string.

    Returns:
        bool: True if the user existed and was deleted, False otherwise.
    """
    if not isinstance(users, dict):
        return False

    if user_id is None:
        return False

    uid = str(user_id)
    if uid not in users:
        return False

    try:
        del users[uid]
        return True
    except Exception:
        return False


def delete_user_account(request: dict) -> bool:
    """
    Deletes a user account in a web application.

    Expects the following globals to be available:
    - database: dict
    - EXPECTED_CSRF_TOKEN: str

    Args:
        request (dict): The request object containing:
            - headers (dict): CSRF token should be present (e.g., 'X-CSRF-Token').
            - body (dict): Must contain 'user_id'.

    Returns:
        bool: True if the user is deleted successfully, False otherwise.
    """
    # Basic request validation
    if not isinstance(request, dict):
        logger.warning("Invalid request type; expected dict, got %s", type(request).__name__)
        return False

    headers = request.get("headers")
    body = request.get("body")

    if not isinstance(headers, dict) or not isinstance(body, dict):
        logger.warning(
            "Invalid request structure; headers type=%s, body type=%s",
            type(headers).__name__ if headers is not None else "None",
            type(body).__name__ if body is not None else "None",
        )
        return False

    # Extract user_id early for logging purposes (no action performed yet)
    user_id = body.get("user_id")
    user_id_str = str(user_id) if user_id is not None else None
    logger.info("Attempting to delete user; user_id=%s", user_id_str)

    # Validate CSRF token using a pure helper with expected token from globals
    expected_token = globals().get("EXPECTED_CSRF_TOKEN")
    if not isinstance(expected_token, str):
        logger.error(
            "EXPECTED_CSRF_TOKEN not configured correctly; expected str, got %s",
            type(expected_token).__name__,
        )
        return False

    csrf_valid = validate_csrf_token(headers, expected_token)
    logger.info("CSRF validation result for user_id=%s: %s", user_id_str, "valid" if csrf_valid else "invalid")
    if not csrf_valid:
        logger.warning("Rejecting deletion due to invalid CSRF token; user_id=%s", user_id_str)
        return False

    # Extract and validate user_id
    if user_id is None:
        logger.warning("Missing user_id in request body; cannot perform deletion")
        return False

    # Access the simulated database via globals
    db = globals().get("database")
    if not isinstance(db, dict):
        logger.error("Database not available or invalid; type=%s", type(db).__name__)
        return False

    users = db.get("users")
    if not isinstance(users, dict):
        logger.error("Database 'users' mapping not available or invalid; type=%s", type(users).__name__)
        return False

    # Perform deletion using a pure helper
    deletion_result = delete_user_from_users(users, user_id)
    uid = str(user_id)
    if deletion_result:
        logger.info("User deletion successful; user_id=%s", uid)
    else:
        if uid not in users:
            logger.info("User not found; no deletion performed; user_id=%s", uid)
        else:
            logger.error("User deletion failed due to an unexpected error; user_id=%s", uid)

    return deletion_result
