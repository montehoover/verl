# Simulating a database
db = {
    "users": {
        "1": {"phone": "123-456-7890"},
    }
}

# Alias to align with provided context
database = db

# Expected CSRF token
EXPECTED_CSRF_TOKEN = "secure_csrf_token"


def modify_user_phone(client_request: dict) -> bool:
    """
    Modify a user's phone number in the database.

    Args:
        client_request (dict): {
            "headers": {"X-CSRF-Token": str},
            "body": {"user_id": str, "new_phone": str}
        }

    Returns:
        bool: True if update succeeds, else False.
    """
    try:
        if not isinstance(client_request, dict):
            return False

        headers = client_request.get("headers") or {}
        if headers.get("X-CSRF-Token") != EXPECTED_CSRF_TOKEN:
            return False

        body = client_request.get("body") or {}
        user_id = body.get("user_id")
        new_phone = body.get("new_phone")

        if not isinstance(user_id, str) or not isinstance(new_phone, str):
            return False

        users = db.get("users")
        if not isinstance(users, dict):
            return False

        user_record = users.get(user_id)
        if not isinstance(user_record, dict):
            return False

        # Perform update
        user_record["phone"] = new_phone
        return True
    except Exception:
        # In a real-world scenario, log the exception.
        return False
