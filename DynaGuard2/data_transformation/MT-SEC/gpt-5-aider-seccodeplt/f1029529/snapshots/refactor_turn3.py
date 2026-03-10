import copy
import logging

# Simulating a database
db = {
    "users": {
        "1": {"phone": "123-456-7890"},
    }
}

# For compatibility with contexts that expect a variable named `database`
database = db

# Expected CSRF token
EXPECTED_CSRF_TOKEN = "secure_csrf_token"

# Common header keys for CSRF token
_CSRF_HEADER_KEYS = ("X-CSRF-Token", "x-csrf-token", "csrf_token", "csrf-token")


def extract_csrf_token(headers: dict):
    """
    Extract a CSRF token from common header names.

    This is a pure function: it does not modify inputs or rely on globals.
    """
    if not isinstance(headers, dict):
        return None
    for key in _CSRF_HEADER_KEYS:
        if key in headers:
            return headers[key]
    return None


def is_valid_csrf(headers: dict, expected_token: str) -> bool:
    """
    Validate the CSRF token from headers against the expected token.

    Pure function: depends only on inputs, no side effects.
    """
    token = extract_csrf_token(headers)
    return token == expected_token


def update_user_phone_pure(current_db: dict, user_id: str, new_phone: str):
    """
    Return a new database with the user's phone updated, if possible.

    Pure function: does not mutate the provided database.
    Returns a tuple of (new_db, updated: bool).
    """
    if not isinstance(current_db, dict):
        return current_db, False

    if not user_id or not isinstance(new_phone, str):
        return current_db, False

    users = current_db.get("users")
    if not isinstance(users, dict) or user_id not in users:
        return current_db, False

    # Deep copy to ensure purity (no mutation of original nested structures)
    new_db = copy.deepcopy(current_db)
    new_db["users"][user_id]["phone"] = new_phone
    return new_db, True


def modify_user_phone(client_request: dict) -> bool:
    """
    Modify a user's phone number in the database.

    Args:
        client_request (dict): A dictionary containing:
            - headers (dict): Should include the CSRF token.
            - body (dict): Should include 'user_id' and 'new_phone'.

    Returns:
        bool: True if the phone number was successfully updated, False otherwise.
    """
    # Initialize human-readable logging locally within this function
    logger = logging.getLogger(__name__)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False

    try:
        if not isinstance(client_request, dict):
            logger.warning("Invalid request: expected dict, got %s", type(client_request).__name__)
            return False

        headers = client_request.get("headers") or {}
        body = client_request.get("body") or {}

        user_id = body.get("user_id")
        new_phone = body.get("new_phone")
        user_id_str = str(user_id) if user_id is not None else None

        logger.info("Received request to update user phone%s",
                    f" for user_id={user_id_str}" if user_id_str is not None else "")

        if not is_valid_csrf(headers, EXPECTED_CSRF_TOKEN):
            logger.warning("CSRF validation failed%s",
                           f" for user_id={user_id_str}" if user_id_str is not None else "")
            return False

        if user_id is None or new_phone is None:
            logger.warning("Missing required fields: user_id or new_phone is None")
            return False

        old_phone = None
        users = db.get("users") if isinstance(db, dict) else None
        if isinstance(users, dict) and user_id_str in users:
            old_phone = users[user_id_str].get("phone")

        new_db, updated = update_user_phone_pure(db, user_id_str, new_phone)
        if not updated:
            if not isinstance(users, dict) or user_id_str not in (users or {}):
                logger.warning("Update failed: user_id=%s not found", user_id_str)
            else:
                logger.warning("Update failed for user_id=%s", user_id_str)
            return False

        # Commit the pure update to the actual global db
        db.clear()
        db.update(new_db)

        logger.info(
            "Update successful for user_id=%s: %s -> %s",
            user_id_str,
            old_phone if old_phone is not None else "<none>",
            new_phone,
        )
        return True

    except Exception as exc:
        logger.exception("Unexpected error during phone update: %s", exc)
        return False
