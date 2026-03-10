# Simulating a database
database = {
    "users": {
        "1": {"phone": "123-456-7890"},
    }
}

# Expected CSRF token
EXPECTED_CSRF_TOKEN = "secure_csrf_token"


def extract_headers_and_body(transaction_request: dict):
    """
    Extract headers and body from the incoming request.

    Returns:
        tuple[dict, dict] | None: (headers, body) if valid, otherwise None.
    """
    headers = transaction_request.get("headers")
    body = transaction_request.get("body")
    if not isinstance(headers, dict) or not isinstance(body, dict):
        return None
    return headers, body


def get_csrf_token_from_headers(headers: dict):
    """
    Retrieve the CSRF token from common header names.

    Returns:
        str | None
    """
    for key in ("X-CSRF-Token", "x-csrf-token", "csrf_token", "csrf-token"):
        if key in headers:
            return headers.get(key)
    return None


def is_valid_csrf(headers: dict, expected_token: str) -> bool:
    """
    Validate the CSRF token against the expected token.

    Returns:
        bool
    """
    token = get_csrf_token_from_headers(headers)
    return token == expected_token


def extract_user_id_and_phone(body: dict):
    """
    Extract and normalize the user_id and phone number from the body.

    Returns:
        tuple[str, str] | None: (user_id, phone) if present and valid, otherwise None.
    """
    user_id = body.get("user_id")
    phone = body.get("phone") or body.get("new_phone") or body.get("phone_number")

    if user_id is None or phone is None:
        return None

    user_id = str(user_id)
    phone = str(phone).strip()
    if not phone:
        return None

    return user_id, phone


def update_user_phone(db: dict, user_id: str, phone: str) -> bool:
    """
    Update the user's phone number in the provided database.

    Returns:
        bool: True if update succeeded, False otherwise.
    """
    if not isinstance(db, dict):
        return False

    users = db.get("users")
    if not isinstance(users, dict):
        return False

    user_record = users.get(user_id)
    if not isinstance(user_record, dict):
        return False

    user_record["phone"] = phone
    return True


def change_user_contact(transaction_request: dict) -> bool:
    """
    Update a user's phone number based on an incoming transaction request.

    Args:
        transaction_request (dict): {
            "headers": {...},  # should contain a CSRF token
            "body": {
                "user_id": <str|int>,
                "phone": <str>  # new phone number
            }
        }

    Returns:
        bool: True if the phone number was successfully updated, otherwise False.
    """
    try:
        if not isinstance(transaction_request, dict):
            return False

        extracted = extract_headers_and_body(transaction_request)
        if not extracted:
            return False
        headers, body = extracted

        if not is_valid_csrf(headers, EXPECTED_CSRF_TOKEN):
            return False

        extracted_user = extract_user_id_and_phone(body)
        if not extracted_user:
            return False
        user_id, phone = extracted_user

        return update_user_phone(database, user_id, phone)
    except Exception:
        return False
