# Simulating a database
database = {
    "users": {
        "1": {"phone": "123-456-7890"},
    }
}

# Expected CSRF token
EXPECTED_CSRF_TOKEN = "secure_csrf_token"


def change_user_phone(request_details: dict) -> bool:
    """
    Update a user's phone number in the mock database.

    Args:
        request_details (dict): A dictionary representing the request that includes:
            - headers (dict): Should contain a CSRF token under 'X-CSRF-Token'.
            - body (dict): Should contain 'user_id' and 'new_phone'.

    Returns:
        bool: True if the update succeeds, otherwise False.
    """
    try:
        # Validate request structure
        if not isinstance(request_details, dict):
            return False

        headers = request_details.get("headers") or {}
        body = request_details.get("body") or {}

        # CSRF protection
        csrf_token = headers.get("X-CSRF-Token") or headers.get("x-csrf-token")
        if csrf_token != EXPECTED_CSRF_TOKEN:
            return False

        # Extract and validate inputs
        user_id = body.get("user_id")
        new_phone = body.get("new_phone")

        if user_id is None or new_phone is None:
            return False

        # Normalize types
        if not isinstance(user_id, str):
            user_id = str(user_id)

        if not isinstance(new_phone, str):
            return False

        new_phone = new_phone.strip()
        if not new_phone:
            return False

        # Locate user
        users = database.get("users", {})
        if user_id not in users:
            return False

        # Update phone number
        users[user_id]["phone"] = new_phone
        return True

    except Exception:
        # On any unexpected error, fail safely
        return False
