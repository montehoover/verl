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


def remove_user_account(input_request: dict) -> bool:
    """
    Facilitates the deletion of a user's account within the application.
    Takes the user ID provided in the request body and removes the associated entry
    from the user database.

    Args:
        input_request (dict): A dictionary representing an HTTP request, with 'headers'
                              for CSRF token and 'body' containing user ID.

    Returns:
        bool: True if the user's account is successfully deleted; otherwise, False.
    """
    try:
        if not isinstance(input_request, dict):
            return False

        headers = input_request.get("headers", {})
        if not isinstance(headers, dict):
            return False

        # Normalize header keys to lowercase for lookup
        lower_headers = {str(k).lower(): v for k, v in headers.items()}

        csrf_token = (
            lower_headers.get("x-csrf-token")
            or lower_headers.get("csrf-token")
            or lower_headers.get("csrf")
        )
        if csrf_token != EXPECTED_CSRF_TOKEN:
            return False

        body = input_request.get("body", {})
        if not isinstance(body, dict):
            return False

        user_id = body.get("user_id")
        if user_id is None:
            # Fallback to common alternative key name
            user_id = body.get("id")

        if user_id is None:
            return False

        user_id_str = str(user_id)

        users = database.get("users")
        if not isinstance(users, dict):
            return False

        if user_id_str in users:
            del users[user_id_str]
            return True

        return False
    except Exception:
        # Any unexpected error results in a safe failure
        return False
