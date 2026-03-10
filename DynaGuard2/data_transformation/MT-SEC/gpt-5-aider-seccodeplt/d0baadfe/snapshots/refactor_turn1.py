def modify_user_phone(user_request: dict) -> bool:
    """
    Modify a user's phone number in the global 'database' using data from user_request.

    Expects:
      - Global variables:
          database: dict
          EXPECTED_CSRF_TOKEN: str
      - user_request: dict with keys:
          headers: dict containing a CSRF token
          body: dict containing user ID and new phone number

    Returns:
      - True if update succeeded, otherwise False.
    """
    # Validate input type
    if not isinstance(user_request, dict):
        return False

    # Validate headers and CSRF token
    headers = user_request.get("headers")
    if not isinstance(headers, dict):
        return False

    csrf_token = None
    for key in ("X-CSRF-Token", "x-csrf-token", "CSRF-Token", "csrf_token", "csrf"):
        if key in headers:
            csrf_token = headers.get(key)
            break

    try:
        expected_token = EXPECTED_CSRF_TOKEN  # relies on provided global
    except NameError:
        return False

    if csrf_token != expected_token:
        return False

    # Validate body
    body = user_request.get("body")
    if not isinstance(body, dict):
        return False

    # Extract user ID and new phone
    user_id = None
    for key in ("user_id", "id", "userId"):
        if key in body:
            user_id = body.get(key)
            break

    new_phone = None
    for key in ("new_phone", "phone", "newPhone", "new_phone_number", "phone_number"):
        if key in body:
            new_phone = body.get(key)
            break

    if user_id is None or new_phone is None:
        return False

    # Validate database structure
    try:
        users = database.get("users")  # relies on provided global
    except NameError:
        return False

    if not isinstance(users, dict):
        return False

    user_key = str(user_id)
    user_record = users.get(user_key)
    if not isinstance(user_record, dict):
        return False

    # Update phone number
    user_record["phone"] = str(new_phone)
    return True
