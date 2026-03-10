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

    Parameters:
        input_request (dict): A dictionary representing an HTTP request,
            with 'headers' for CSRF token and 'body' containing 'user_id'.

    Returns:
        bool: True if the user's account is successfully deleted; otherwise, False.
    """
    # Validate input structure
    if not isinstance(input_request, dict):
        return False

    headers = input_request.get("headers")
    body = input_request.get("body")

    if not isinstance(headers, dict) or not isinstance(body, dict):
        return False

    # CSRF validation
    csrf_token = headers.get("X-CSRF-Token")
    if csrf_token != EXPECTED_CSRF_TOKEN:
        return False

    # Extract and validate user_id
    user_id = body.get("user_id")
    if user_id is None:
        return False

    # Convert to string to match database key types
    user_id_str = str(user_id)

    # Perform deletion if user exists
    users = database.get("users")
    if not isinstance(users, dict):
        return False

    if user_id_str in users:
        del users[user_id_str]
        return True

    return False
