# Simulating a database
email_database = {
    "users": {
        "1": {"email": "old.email@example.com"},
    }
}

# Expected CSRF token
EXPECTED_CSRF_TOKEN = "secure_csrf_token"


def _get_headers(user_request: dict):
    headers = user_request.get("headers")
    return headers if isinstance(headers, dict) else None


def _get_body(user_request: dict):
    body = user_request.get("body")
    return body if isinstance(body, dict) else None


def _valid_csrf(headers: dict) -> bool:
    return headers.get("X-CSRF-Token") == EXPECTED_CSRF_TOKEN


def _extract_user_id_and_email(body: dict):
    user_id = body.get("user_id")
    new_email = body.get("new_email")

    if user_id is None or new_email is None:
        return None, None

    # Normalize types
    user_id = str(user_id)

    if not isinstance(new_email, str):
        return None, None

    email_value = new_email.strip()
    # Very basic email sanity check
    if not email_value or "@" not in email_value or email_value.startswith("@") or email_value.endswith("@"):
        return None, None

    return user_id, email_value


def _get_user_record(user_id: str):
    users = email_database.get("users")
    if not isinstance(users, dict):
        return None
    record = users.get(user_id)
    return record if isinstance(record, dict) else None


def modify_user_email_address(user_request: dict) -> bool:
    """
    Modify a user's email address.

    Args:
        user_request (dict): The incoming request object containing:
            - headers (dict): Should include 'X-CSRF-Token'
            - body (dict): Should include 'user_id' and 'new_email'
    Returns:
        bool: True if the email address was updated successfully, otherwise False.
    """
    try:
        if not isinstance(user_request, dict):
            return False

        headers = _get_headers(user_request)
        if not headers:
            return False

        if not _valid_csrf(headers):
            return False

        body = _get_body(user_request)
        if not body:
            return False

        user_id, email_value = _extract_user_id_and_email(body)
        if not user_id or not email_value:
            return False

        user_record = _get_user_record(user_id)
        if not user_record:
            return False

        user_record["email"] = email_value
        return True
    except Exception:
        # On any unexpected error, do not update and return False
        return False
