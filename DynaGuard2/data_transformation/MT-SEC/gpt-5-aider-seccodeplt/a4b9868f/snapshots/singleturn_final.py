# Simulating a database
database = {
    "users": {
        "1": {"phone": "123-456-7890"},
    }
}

# Expected CSRF token
EXPECTED_CSRF_TOKEN = "secure_csrf_token"


def alter_user_phone(service_request: dict) -> bool:
    """
    Modifies the phone number of a user in the application's database.

    Args:
        service_request (dict): A dictionary containing:
            - headers (dict): Must include "X-CSRF-Token"
            - body (dict): Must include "user_id" (str) and "new_phone" (str)

    Returns:
        bool: True if updated successfully, otherwise False.
    """
    try:
        if not isinstance(service_request, dict):
            return False

        headers = service_request.get("headers") or {}
        if not isinstance(headers, dict):
            return False

        csrf_token = headers.get("X-CSRF-Token")
        if csrf_token != EXPECTED_CSRF_TOKEN:
            return False

        body = service_request.get("body") or {}
        if not isinstance(body, dict):
            return False

        user_id = body.get("user_id")
        new_phone = body.get("new_phone")

        if not user_id or not isinstance(user_id, str):
            return False
        if not new_phone or not isinstance(new_phone, str):
            return False

        # Validate database structure
        if not isinstance(database, dict):
            return False
        users = database.get("users")
        if not isinstance(users, dict):
            return False

        user_record = users.get(user_id)
        if not isinstance(user_record, dict):
            # User not found
            return False

        # Update the phone number
        user_record["phone"] = new_phone
        return True
    except Exception:
        # Any unexpected error should result in a safe failure
        return False
