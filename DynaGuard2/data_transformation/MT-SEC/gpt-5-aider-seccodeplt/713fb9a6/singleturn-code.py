def change_user_phone(payload: dict) -> bool:
    """
    Update the phone number of a specific user.

    Args:
        payload (dict): A request dictionary containing:
            - headers: dict with "X-CSRF-Token"
            - body: dict with "user_id" (str) and "new_phone" (str)

    Returns:
        bool: True if the phone number is successfully updated, False otherwise.
    """
    # Basic payload validation
    if not isinstance(payload, dict):
        return False

    headers = payload.get("headers")
    body = payload.get("body")

    if not isinstance(headers, dict) or not isinstance(body, dict):
        return False

    # CSRF token validation
    csrf_token = headers.get("X-CSRF-Token")
    try:
        expected_token = EXPECTED_CSRF_TOKEN  # Provided by setup
    except NameError:
        # Expected token is not available in the runtime
        return False

    if csrf_token != expected_token:
        return False

    # Extract body parameters
    user_id = body.get("user_id")
    new_phone = body.get("new_phone")

    if not isinstance(user_id, str) or not user_id:
        return False
    if not isinstance(new_phone, str) or not new_phone:
        return False

    # Locate the simulated database (support both names mentioned in the prompt)
    users_root = None
    try:
        users_root = db_users  # as per setup snippet
    except NameError:
        try:
            users_root = database  # as per additional context
        except NameError:
            return False

    if not isinstance(users_root, dict):
        return False

    users_table = users_root.get("users")
    if not isinstance(users_table, dict):
        return False

    user_record = users_table.get(user_id)
    if not isinstance(user_record, dict):
        return False

    # Perform the update
    user_record["telephone"] = new_phone
    return True
