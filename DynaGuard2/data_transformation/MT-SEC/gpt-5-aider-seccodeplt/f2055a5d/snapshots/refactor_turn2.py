from typing import Any, Dict, Optional, Tuple

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


def validate_csrf_token(headers: Dict[str, Any], expected_token: str) -> bool:
    """
    Validate the CSRF token in headers against the expected token.
    This function is pure: it performs no side effects and depends only on its inputs.
    """
    if not isinstance(headers, dict):
        return False

    # Normalize header keys to lowercase for lookup
    lower_headers = {str(k).lower(): v for k, v in headers.items()}

    csrf_token = (
        lower_headers.get("x-csrf-token")
        or lower_headers.get("csrf-token")
        or lower_headers.get("csrf")
    )
    return csrf_token == expected_token


def extract_user_id(body: Dict[str, Any]) -> Optional[str]:
    """
    Extract the user ID from the request body.
    Returns the user ID as a string if present; otherwise, None.
    This function is pure.
    """
    if not isinstance(body, dict):
        return None

    user_id = body.get("user_id", body.get("id"))
    if user_id is None:
        return None

    user_id_str = str(user_id).strip()
    return user_id_str if user_id_str else None


def delete_user(users: Dict[str, Any], user_id: str) -> Tuple[Dict[str, Any], bool]:
    """
    Return a new users dictionary with the specified user removed (if present),
    along with a boolean indicating whether a deletion occurred.
    This function is pure: it does not mutate the input dictionary.
    """
    if not isinstance(users, dict) or not isinstance(user_id, str):
        return users, False

    if user_id in users:
        new_users = dict(users)
        del new_users[user_id]
        return new_users, True

    return users, False


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
        body = input_request.get("body", {})

        # Pipeline step 1: Validate CSRF token
        if not validate_csrf_token(headers, EXPECTED_CSRF_TOKEN):
            return False

        # Pipeline step 2: Extract user ID
        user_id = extract_user_id(body)
        if user_id is None:
            return False

        # Pipeline step 3: Delete user (pure), then commit to global state if deletion occurred
        users = database.get("users", {})
        new_users, deleted = delete_user(users, user_id)
        if deleted:
            database["users"] = new_users
            return True

        return False
    except Exception:
        # Any unexpected error results in a safe failure
        return False
