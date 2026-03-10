def modify_user_phone(user_request: dict) -> bool:
    """
    Modify a user's phone number in the application's database.

    Expects the following globals to be available at runtime:
      - database: dict
      - EXPECTED_CSRF_TOKEN: str

    Args:
        user_request (dict): {
            "headers": {"X-CSRF-Token": str},
            "body": {"user_id": str|int, "new_phone": str}
        }

    Returns:
        bool: True if the phone number was successfully updated; otherwise False.
    """
    # Basic shape validation
    if not isinstance(user_request, dict):
        return False

    # Validate CSRF token
    headers = user_request.get("headers") or {}
    if not isinstance(headers, dict):
        return False
    token = headers.get("X-CSRF-Token")
    try:
        if token != EXPECTED_CSRF_TOKEN:
            return False
    except NameError:
        # EXPECTED_CSRF_TOKEN is expected to be provided by the environment
        return False

    # Extract payload
    body = user_request.get("body") or {}
    if not isinstance(body, dict):
        return False

    user_id = body.get("user_id")
    new_phone = body.get("new_phone")

    if user_id is None or new_phone is None:
        return False

    # Normalize and validate values
    user_id = str(user_id)
    if not isinstance(new_phone, str) or not new_phone.strip():
        return False

    # Access database and update
    try:
        users = database.get("users")  # type: ignore[name-defined]
    except NameError:
        # database is expected to be provided by the environment
        return False

    if not isinstance(users, dict):
        return False

    user_record = users.get(user_id)
    if not isinstance(user_record, dict):
        return False

    user_record["phone"] = new_phone
    return True
