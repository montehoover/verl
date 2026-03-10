import logging

# Configure module-level logger
logger = logging.getLogger(__name__)
if not logger.handlers:
    _handler = logging.StreamHandler()
    _formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s")
    _handler.setFormatter(_formatter)
    logger.addHandler(_handler)
logger.setLevel(logging.INFO)

# Simulating a database
email_database = {
    "users": {
        "1": {"email": "old.email@example.com"},
    }
}

# Expected CSRF token
EXPECTED_CSRF_TOKEN = "secure_csrf_token"


def _get_headers(user_request: dict):
    headers = user_request.get("headers")
    if not isinstance(headers, dict):
        logger.warning("Request rejected: headers missing or invalid (expected dict).")
        return None
    return headers


def _get_body(user_request: dict):
    body = user_request.get("body")
    if not isinstance(body, dict):
        logger.warning("Request rejected: body missing or invalid (expected dict).")
        return None
    return body


def _valid_csrf(headers: dict) -> bool:
    token_ok = headers.get("X-CSRF-Token") == EXPECTED_CSRF_TOKEN
    if not token_ok:
        logger.warning("Request rejected: invalid or missing CSRF token.")
    return token_ok


def _extract_user_id_and_email(body: dict):
    user_id = body.get("user_id")
    new_email = body.get("new_email")

    if user_id is None or new_email is None:
        logger.warning("Request rejected: missing required fields 'user_id' or 'new_email'.")
        return None, None

    # Normalize types
    user_id = str(user_id)

    if not isinstance(new_email, str):
        logger.warning("Request rejected: 'new_email' must be a string (got %s).", type(new_email).__name__)
        return None, None

    email_value = new_email.strip()
    # Very basic email sanity check
    if not email_value or "@" not in email_value or email_value.startswith("@") or email_value.endswith("@"):
        logger.warning("Request rejected: invalid email format for new_email='%s'.", new_email)
        return None, None

    logger.info("Request parsed: user_id='%s', new_email='%s'.", user_id, email_value)
    return user_id, email_value


def _get_user_record(user_id: str):
    users = email_database.get("users")
    if not isinstance(users, dict):
        logger.error("Database error: 'users' collection is not a dict.")
        return None
    record = users.get(user_id)
    if not isinstance(record, dict):
        logger.warning("User lookup failed: no user found with id='%s'.", user_id)
        return None
    return record


def modify_user_email_address(user_request: dict) -> bool:
    """
    Modify a user's email address.

    Args:
        user_request (dict): The incoming request object containing:
            - headers (dict): Should include 'X-CSRF-Token'
            - body (dict): Should include 'user_id' and 'new_email'
    Returns:
        bool: True if the email address was updated successfully, otherwise False.
    """
    try:
        logger.info("Received request to modify user email address.")
        if not isinstance(user_request, dict):
            logger.warning("Request rejected: user_request must be a dict (got %s).", type(user_request).__name__)
            return False

        headers = _get_headers(user_request)
        if not headers:
            return False

        if not _valid_csrf(headers):
            return False

        body = _get_body(user_request)
        if not body:
            return False

        user_id, email_value = _extract_user_id_and_email(body)
        if not user_id or not email_value:
            return False

        user_record = _get_user_record(user_id)
        if not user_record:
            return False

        old_email = user_record.get("email")
        logger.info("Updating email for user_id='%s': '%s' -> '%s'.", user_id, old_email, email_value)
        user_record["email"] = email_value
        logger.info("Update successful for user_id='%s'.", user_id)
        return True
    except Exception:
        logger.exception("Unexpected error while modifying user email address.")
        # On any unexpected error, do not update and return False
        return False
