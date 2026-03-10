# Simulating a database
email_database = {
    "users": {
        "1": {"email": "old.email@example.com"},
    }
}

def validate_csrf(headers: dict) -> bool:
    """
    Validates a CSRF token from request headers.

    Args:
        headers: A dictionary representing the request headers.
                 Expected to contain 'X-CSRF-Token'.

    Returns:
        True if the CSRF token is valid, False otherwise.
    """
    expected_token = 'secure_csrf_token'
    actual_token = headers.get('X-CSRF-Token')
    return actual_token == expected_token

def fetch_user_by_id(user_id: str) -> dict | None:
    """
    Retrieves user information by user ID from the email_database.

    Args:
        user_id: The ID of the user to retrieve.

    Returns:
        A dictionary containing the user's data if found, otherwise None.
    """
    return email_database["users"].get(user_id)

def modify_user_email_address(user_request: dict) -> bool:
    """
    Updates a user's email address after validating CSRF token and user existence.

    Args:
        user_request: A dictionary containing 'headers' and 'body'.
                      'headers' should contain 'X-CSRF-Token'.
                      'body' should contain 'user_id' and 'new_email'.

    Returns:
        True if the email was updated successfully, False otherwise.
    """
    if not validate_csrf(user_request.get('headers', {})):
        return False

    body = user_request.get('body', {})
    user_id = body.get('user_id')
    new_email = body.get('new_email')

    if not user_id or not new_email:
        return False

    user = fetch_user_by_id(user_id)
    if user is None:
        return False

    email_database['users'][user_id]['email'] = new_email
    return True
