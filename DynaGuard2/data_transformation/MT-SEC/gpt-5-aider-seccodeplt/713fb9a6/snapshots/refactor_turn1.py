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
        return False

    headers = payload.get("headers") or {}
    body = payload.get("body") or {}

    # Extract CSRF token from common header keys
    token = None
    for key in ("X-CSRF-Token", "x-csrf-token", "csrf_token", "csrf", "X_CSRF_Token"):
        if key in headers:
            token = headers.get(key)
            break

    if token != EXPECTED_CSRF_TOKEN:
        return False

    # Extract user_id
    user_id = None
    for key in ("user_id", "id", "userId"):
        if key in body:
            user_id = str(body.get(key))
            break

    if not user_id:
        return False

    # Extract new phone number
    new_phone = None
    for key in ("phone", "telephone", "new_phone", "new_phone_number"):
        if key in body:
            new_phone = body.get(key)
            break

    if not isinstance(new_phone, str) or not new_phone:
        return False

    # Update database
    try:
        users = db_users.get("users", {})
        if user_id not in users:
            return False

        users[user_id]["telephone"] = new_phone
        return True
    except Exception:
        return False
