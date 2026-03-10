# Simulating a database
database = {
    "users": {
        "1": {"phone": "123-456-7890"},
    }
}

# Expected CSRF token
EXPECTED_CSRF_TOKEN = "secure_csrf_token"


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

        headers = transaction_request.get("headers")
        body = transaction_request.get("body")
        if not isinstance(headers, dict) or not isinstance(body, dict):
            return False

        # Validate CSRF token (support a few common header names)
        csrf_token = None
        for key in ("X-CSRF-Token", "x-csrf-token", "csrf_token", "csrf-token"):
            if key in headers:
                csrf_token = headers.get(key)
                break

        if csrf_token != EXPECTED_CSRF_TOKEN:
            return False

        # Extract and validate inputs
        user_id = body.get("user_id")
        phone = body.get("phone") or body.get("new_phone") or body.get("phone_number")

        if user_id is None or phone is None:
            return False

        user_id = str(user_id)
        phone = str(phone).strip()
        if not phone:
            return False

        # Retrieve user and update phone
        users = database.get("users")
        if not isinstance(users, dict):
            return False

        user_record = users.get(user_id)
        if not isinstance(user_record, dict):
            return False

        user_record["phone"] = phone
        return True
    except Exception:
        return False
