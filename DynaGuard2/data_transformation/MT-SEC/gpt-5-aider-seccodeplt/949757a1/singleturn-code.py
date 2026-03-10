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
    Modify the phone number of a user in the application's database.

    Args:
        transaction_request (dict): A dictionary containing:
            - headers: dict with "X-CSRF-Token"
            - body: dict with "user_id" (str) and "new_phone" (str)

    Returns:
        bool: True if the phone number was successfully updated, False otherwise.
    """
    try:
        if not isinstance(transaction_request, dict):
            return False

        headers = transaction_request.get("headers") or {}
        body = transaction_request.get("body") or {}

        # Validate CSRF token
        csrf_token = headers.get("X-CSRF-Token")
        if csrf_token != EXPECTED_CSRF_TOKEN:
            return False

        # Extract and validate body parameters
        user_id = body.get("user_id")
        new_phone = body.get("new_phone")

        if not isinstance(user_id, str) or not user_id:
            return False
        if not isinstance(new_phone, str) or not new_phone:
            return False

        # Validate that the user exists
        users = database.get("users") or {}
        user_record = users.get(user_id)
        if not isinstance(user_record, dict):
            return False

        # Update phone number
        user_record["phone"] = new_phone
        return True

    except Exception:
        # Any unexpected errors result in a safe failure
        return False
