# Simulating a database
database = {
    "users": {
        "1": {"phone": "123-456-7890"},
    }
}

# Expected CSRF token
EXPECTED_CSRF_TOKEN = "secure_csrf_token"


def validate_csrf_token(headers: dict, expected_token: str) -> bool:
    """
    Pure function that validates a CSRF token from request headers.

    Args:
        headers (dict): Request headers that may contain the CSRF token.
        expected_token (str): The expected CSRF token value.

    Returns:
        bool: True if the token is valid, False otherwise.
    """
    if not isinstance(headers, dict):
        return False

    # Normalize header keys to lowercase for case-insensitive access
    lower_headers = {str(k).lower(): v for k, v in headers.items()}
    token = lower_headers.get("x-csrf-token")
    return token == expected_token


def updated_users_with_phone(users: dict, user_id: str, new_phone: str):
    """
    Pure function that returns a new 'users' mapping with the updated phone number
    for the specified user_id. Returns None if the user does not exist or inputs are invalid.

    Args:
        users (dict): Original users mapping.
        user_id (str): The user's ID to update.
        new_phone (str): The new phone number.

    Returns:
        dict | None: A new users mapping with the updated phone, or None if not applicable.
    """
    if not isinstance(users, dict):
        return None

    if not isinstance(user_id, str):
        user_id = str(user_id)

    if not isinstance(new_phone, str):
        return None

    new_phone = new_phone.strip()
    if not new_phone:
        return None

    if user_id not in users or not isinstance(users[user_id], dict):
        return None

    # Create a shallow copy of users and the specific user entry to avoid side effects
    new_users = dict(users)
    updated_user = dict(users[user_id])
    updated_user["phone"] = new_phone
    new_users[user_id] = updated_user
    return new_users


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
        if not validate_csrf_token(headers, EXPECTED_CSRF_TOKEN):
            return False

        # Extract and validate inputs
        user_id = body.get("user_id")
        new_phone = body.get("new_phone")

        if user_id is None or new_phone is None:
            return False

        # Normalize types where necessary before handing off to pure updater
        if not isinstance(user_id, str):
            user_id = str(user_id)

        # Locate current users mapping
        users = database.get("users", {})
        updated_users = updated_users_with_phone(users, user_id, new_phone)
        if updated_users is None:
            return False

        # Commit the updated mapping to the database
        database["users"] = updated_users
        return True

    except Exception:
        # On any unexpected error, fail safely
        return False
