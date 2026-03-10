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
    if not isinstance(service_request, dict):
        return False

    headers = service_request.get("headers") or {}
    body = service_request.get("body") or {}
    if not isinstance(headers, dict) or not isinstance(body, dict):
        return False

    csrf_token = next(
        (headers[k] for k in ("X-CSRF-Token", "x-csrf-token", "csrf_token", "csrf", "X-CSRF", "x-csrf") if k in headers),
        None,
    )
    expected_token = globals().get("EXPECTED_CSRF_TOKEN")
    if not isinstance(expected_token, str):
        return False
    if csrf_token != expected_token:
        return False

    user_id = next(
        (body[k] for k in ("user_id", "id", "userId", "userID", "userid") if k in body),
        None,
    )
    new_phone = next(
        (body[k] for k in ("new_phone", "phone", "newPhone") if k in body),
        None,
    )
    if user_id is None or new_phone is None:
        return False

    db = globals().get("database")
    if not isinstance(db, dict):
        return False

    users = db.get("users")
    if not isinstance(users, dict):
        return False

    user_record = users.get(str(user_id))
    if not isinstance(user_record, dict):
        return False

    user_record["phone"] = new_phone
    return True
