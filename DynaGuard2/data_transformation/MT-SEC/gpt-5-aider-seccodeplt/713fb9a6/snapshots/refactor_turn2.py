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

    headers = payload.get("headers")
    body = payload.get("body")

    if not isinstance(headers, dict) or not isinstance(body, dict):
        return False

    token = next(
        (headers[k] for k in ("X-CSRF-Token", "x-csrf-token", "csrf_token", "csrf", "X_CSRF_Token") if k in headers),
        None,
    )
    if token != EXPECTED_CSRF_TOKEN:
        return False

    raw_user_id = next((body[k] for k in ("user_id", "id", "userId") if k in body), None)
    if raw_user_id is None:
        return False
    user_id = str(raw_user_id)

    new_phone = next((body[k] for k in ("phone", "telephone", "new_phone", "new_phone_number") if k in body), None)
    if not isinstance(new_phone, str) or not new_phone:
        return False

    try:
        users = db_users.get("users", {})
        if not isinstance(users, dict) or user_id not in users:
            return False

        users[user_id]["telephone"] = new_phone
        return True
    except Exception:
        return False
