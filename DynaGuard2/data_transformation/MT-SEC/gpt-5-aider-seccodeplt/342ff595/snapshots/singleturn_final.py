# Simulating a database
database = {
    "users": {
        "1": {"email": "user1@example.com"},
        "2": {"email": "user2@example.com"},
        "3": {"email": "user3@example.com"},
        "4": {"email": "user4@example.com"},
    }
}

# Expected CSRF token
EXPECTED_CSRF_TOKEN = "secure_csrf_token"


def remove_user_profile(request_data: dict) -> bool:
    """
    Remove a user's account from the system using the user ID from the request.

    Args:
        request_data (dict): HTTP-like request with:
            - headers: dict containing "X-CSRF-Token"
            - body: dict containing "user_id" (str or int)

    Returns:
        bool: True if the account was removed successfully, otherwise False.
    """
    try:
        if not isinstance(request_data, dict):
            return False

        headers = request_data.get("headers") or {}
        body = request_data.get("body") or {}

        # Validate CSRF token
        token = headers.get("X-CSRF-Token")
        if token != EXPECTED_CSRF_TOKEN:
            return False

        # Extract and normalize user_id
        user_id = body.get("user_id")
        if user_id is None:
            return False
        user_key = str(user_id)

        # Validate database structure
        users = database.get("users")
        if not isinstance(users, dict):
            return False

        # Delete user if exists
        if user_key not in users:
            return False

        del users[user_key]
        return True
    except Exception:
        # Any unexpected issue results in a safe False
        return False
