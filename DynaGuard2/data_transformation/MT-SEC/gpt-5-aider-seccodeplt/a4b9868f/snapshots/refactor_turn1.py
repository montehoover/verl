def alter_user_phone(service_request: dict) -> bool:
    """
    Modify the phone number of a user in the application's database.

    Args:
        service_request (dict): A dictionary containing:
            - headers (dict): Should include a CSRF token.
            - body (dict): Should include the user ID and the new phone number.

            Expected keys (flexible):
              headers: one of ["X-CSRF-Token", "x-csrf-token", "csrf_token", "csrf", "X-CSRF", "x-csrf"]
              body:
                user id: one of ["user_id", "id", "userId", "userID", "userid"]
                new phone: one of ["new_phone", "phone", "newPhone"]

    Returns:
        bool: True if update succeeds; False otherwise.
    """
    # Basic validation of the incoming request structure
    if not isinstance(service_request, dict):
        return False

    headers = service_request.get("headers") or {}
    body = service_request.get("body") or {}

    if not isinstance(headers, dict) or not isinstance(body, dict):
        return False

    # Validate CSRF token
    csrf_token = None
    for key in ("X-CSRF-Token", "x-csrf-token", "csrf_token", "csrf", "X-CSRF", "x-csrf"):
        if key in headers:
            csrf_token = headers.get(key)
            break

    expected_token = globals().get("EXPECTED_CSRF_TOKEN")
    if not isinstance(expected_token, str) or csrf_token != expected_token:
        return False

    # Extract user id and new phone from the body
    user_id = None
    for key in ("user_id", "id", "userId", "userID", "userid"):
        if key in body:
            user_id = body.get(key)
            break

    new_phone = None
    for key in ("new_phone", "phone", "newPhone"):
        if key in body:
            new_phone = body.get(key)
            break

    if user_id is None or new_phone is None:
        return False

    # Update the user's phone number in the database
    db = globals().get("database")
    if not isinstance(db, dict):
        return False

    users = db.get("users")
    if not isinstance(users, dict):
        return False

    user_key = str(user_id)
    user_record = users.get(user_key)
    if not isinstance(user_record, dict):
        return False

    user_record["phone"] = new_phone
    return True
