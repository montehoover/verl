# Simulating a database
user_data = {
    "users": {
        "1": {"mobile": "123-456-7891"},
    }
}

# Alias to match possible alternative naming in some contexts
database = user_data

# Expected CSRF token
EXPECTED_CSRF_TOKEN = "known_csrf_token"


def alter_user_phone(input: dict) -> bool:
    """
    Update a user's phone number in the simulated database.

    Args:
        input (dict): Request dictionary containing:
            - headers: dict with "X-CSRF-Token"
            - body: dict with "user_id" and "new_phone"

    Returns:
        bool: True if the phone number was updated, otherwise False.
    """
    try:
        if not isinstance(input, dict):
            return False

        headers = input.get("headers")
        body = input.get("body")

        if not isinstance(headers, dict) or not isinstance(body, dict):
            return False

        # CSRF validation
        if headers.get("X-CSRF-Token") != EXPECTED_CSRF_TOKEN:
            return False

        user_id = body.get("user_id")
        new_phone = body.get("new_phone")

        if not isinstance(user_id, str) or not isinstance(new_phone, str) or not new_phone:
            return False

        users = user_data.get("users")
        if not isinstance(users, dict):
            return False

        user_record = users.get(user_id)
        if not isinstance(user_record, dict):
            return False

        user_record["mobile"] = new_phone
        return True
    except Exception:
        # Any unexpected issue is treated as a failure
        return False
