import logging

# Configure logging (only if no handlers are already configured)
if not logging.getLogger().hasHandlers():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

logger = logging.getLogger(__name__)

# Simulating a database
db_users = {
    "users": {
        "1": {"telephone": "123-556-7890"},
    }
}

# Expected CSRF token
EXPECTED_CSRF_TOKEN = "csrf_token_secured"


def change_user_phone(payload: dict) -> bool:
    """
    Update the phone number for a user.

    Args:
        payload: dict containing:
            - headers: dict with CSRF token
            - body: dict with user ID and new phone number

    Returns:
        True if update succeeds, False otherwise.
    """
    if not isinstance(payload, dict):
        logger.warning("change_user_phone failed: invalid payload type. user_id=%s", None)
        return False

    headers = payload.get("headers")
    body = payload.get("body")

    # Extract attempted user_id early for logging purposes
    attempted_user_id = None
    if isinstance(body, dict):
        attempted_raw_user_id = next((body.get(k) for k in ("user_id", "id", "userId") if k in body), None)
        if attempted_raw_user_id is not None:
            attempted_user_id = str(attempted_raw_user_id)

    if not isinstance(headers, dict) or not isinstance(body, dict):
        logger.warning("change_user_phone failed: headers/body missing or invalid. user_id=%s", attempted_user_id)
        return False

    token = next(
        (headers[k] for k in ("X-CSRF-Token", "x-csrf-token", "csrf_token", "csrf", "X_CSRF_Token") if k in headers),
        None,
    )
    if token != EXPECTED_CSRF_TOKEN:
        logger.warning("change_user_phone failed: invalid CSRF token. user_id=%s", attempted_user_id)
        return False

    raw_user_id = next((body[k] for k in ("user_id", "id", "userId") if k in body), None)
    if raw_user_id is None:
        logger.warning("change_user_phone failed: missing user_id. user_id=%s", attempted_user_id)
        return False
    user_id = str(raw_user_id)

    new_phone = next((body[k] for k in ("phone", "telephone", "new_phone", "new_phone_number") if k in body), None)
    if not isinstance(new_phone, str) or not new_phone:
        logger.warning("change_user_phone failed: invalid new phone. user_id=%s", user_id)
        return False

    try:
        users = db_users.get("users", {})
        if not isinstance(users, dict):
            logger.warning("change_user_phone failed: users store invalid. user_id=%s", user_id)
            return False
        if user_id not in users:
            logger.warning("change_user_phone failed: user not found. user_id=%s", user_id)
            return False

        users[user_id]["telephone"] = new_phone
        logger.info("change_user_phone success: phone updated. user_id=%s", user_id)
        return True
    except Exception:
        logger.exception("change_user_phone failed with exception. user_id=%s", user_id)
        return False
