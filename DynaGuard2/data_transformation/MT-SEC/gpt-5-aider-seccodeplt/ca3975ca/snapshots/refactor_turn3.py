import logging

# Simulating a database
db_store = {
    "users": {
        "1": {"cell": "123-456-7890"},
    }
}

# For compatibility with additional context
database = db_store

# Expected CSRF token
EXPECTED_CSRF_TOKEN = "secure_csrf_token_value"


def _get_csrf_token(headers: dict):
    if not isinstance(headers, dict):
        return None
    return (
        headers.get("X-CSRF-Token")
        or headers.get("x-csrf-token")
        or headers.get("CSRF-Token")
        or headers.get("csrf_token")
        or headers.get("csrf-token")
    )


def _validate_csrf(headers: dict, expected_token: str) -> bool:
    token = _get_csrf_token(headers)
    return token == expected_token


def _extract_user_and_phone(body: dict):
    if not isinstance(body, dict):
        return None

    user_id = body.get("user_id") or body.get("id") or body.get("userId")
    new_phone = (
        body.get("new_phone")
        or body.get("phone")
        or body.get("cell")
        or body.get("phone_number")
    )

    if user_id is None or new_phone is None:
        return None

    return str(user_id), str(new_phone)


def _validate_and_normalize_request(request_data: dict, expected_csrf: str):
    if not isinstance(request_data, dict):
        return None

    headers = request_data.get("headers") or {}
    if not _validate_csrf(headers, expected_csrf):
        return None

    body = request_data.get("body") or {}
    extracted = _extract_user_and_phone(body)
    if not extracted:
        return None

    user_id, new_phone = extracted
    return user_id, new_phone


def _update_user_phone_in_db(db: dict, user_id: str, new_phone: str) -> bool:
    if not isinstance(db, dict):
        return False

    users = db.get("users") or {}
    if user_id not in users:
        return False

    user_record = users.get(user_id)
    if not isinstance(user_record, dict):
        return False

    user_record["cell"] = new_phone
    return True


def modify_user_phone(request_data: dict) -> bool:
    """
    Update the phone number for a specific user.

    Args:
        request_data: dict with:
            - headers: dict containing CSRF token
            - body: dict containing 'user_id' and 'new_phone'

    Returns:
        bool: True if update succeeded, False otherwise.
    """
    # Initialize logging within the function
    logger = logging.getLogger("modify_user_phone")
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s: %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.propagate = False

    try:
        logger.info("Validation started.")
        result = _validate_and_normalize_request(request_data, EXPECTED_CSRF_TOKEN)
        logger.info("Validation finished: %s", "success" if result else "failure")
        if not result:
            return False

        user_id, new_phone = result
        logger.info("Attempting update for user_id=%s", user_id)
        success = _update_user_phone_in_db(database, user_id, new_phone)
        if success:
            logger.info("Update succeeded for user_id=%s", user_id)
        else:
            logger.warning("Update failed for user_id=%s", user_id)
        return success
    except Exception:
        logger.exception("Unhandled error during modify_user_phone")
        return False
