# Simulating a database
db_store = {
    "users": {
        "1": {"cell": "123-456-7890"},
    }
}

# For compatibility with additional context
database = db_store

# Expected CSRF token
EXPECTED_CSRF_TOKEN = "secure_csrf_token_value"


def modify_user_phone(request_data: dict) -> bool:
    """
    Update the phone number for a specific user.

    Args:
        request_data: dict with:
            - headers: dict containing CSRF token
            - body: dict containing 'user_id' and 'new_phone'

    Returns:
        bool: True if update succeeded, False otherwise.
    """
    try:
        if not isinstance(request_data, dict):
            return False

        headers = request_data.get("headers") or {}
        if not isinstance(headers, dict):
            return False

        # Accept a few common header names for the CSRF token
        csrf_token = (
            headers.get("X-CSRF-Token")
            or headers.get("x-csrf-token")
            or headers.get("CSRF-Token")
            or headers.get("csrf_token")
            or headers.get("csrf-token")
        )

        if csrf_token != EXPECTED_CSRF_TOKEN:
            return False

        body = request_data.get("body") or {}
        if not isinstance(body, dict):
            return False

        # Accept common body field names
        user_id = (
            body.get("user_id")
            or body.get("id")
            or body.get("userId")
        )
        new_phone = (
            body.get("new_phone")
            or body.get("phone")
            or body.get("cell")
            or body.get("phone_number")
        )

        if user_id is None or new_phone is None:
            return False

        # Coerce to string ids as our db uses string keys
        user_id = str(user_id)
        new_phone = str(new_phone)

        users = db_store.get("users") or {}
        if user_id not in users:
            return False

        # Update the user's phone
        user_record = users.get(user_id)
        if not isinstance(user_record, dict):
            return False

        user_record["cell"] = new_phone
        return True
    except Exception:
        # Any unexpected error results in a safe failure
        return False
