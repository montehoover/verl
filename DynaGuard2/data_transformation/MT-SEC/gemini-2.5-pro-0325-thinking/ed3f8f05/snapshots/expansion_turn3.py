def verify_csrf_token(request_headers: dict) -> bool:
    """
    Checks whether a CSRF token in a given request headers dictionary
    matches the expected token.

    Args:
        request_headers: A dictionary representing the request headers.

    Returns:
        True if the token matches, False otherwise.
    """
    actual_token = request_headers.get('X-CSRF-Token')
    return actual_token == EXPECTED_CSRF_TOKEN

# Simulating a database
database = {
    "users": {
        "1": {"email": "old.email@example.com"},
    }
}

# Expected CSRF token
EXPECTED_CSRF_TOKEN = "secure_csrf_token"


def get_user_information(user_id: str) -> dict | None:
    """
    Retrieves a user's profile from the mock database.

    Args:
        user_id: The ID of the user to retrieve.

    Returns:
        The user's data if they exist, or None if not.
    """
    return database.get("users", {}).get(user_id)

def change_user_email(request_data: dict) -> bool:
    """
    Updates a user's email address based on incoming HTTP request data.

    Args:
        request_data: A dictionary containing 'headers' and 'body'.
                      The 'body' should contain 'user_id' and 'new_email'.

    Returns:
        True if the email is successfully updated, False otherwise.
    """
    headers = request_data.get('headers', {})
    body = request_data.get('body', {})

    if not verify_csrf_token(headers):
        return False

    user_id = body.get('user_id')
    new_email = body.get('new_email')

    if not user_id or not new_email:
        return False

    user_data = get_user_information(user_id)
    if user_data is None:
        return False

    # Update the email in the database
    database["users"][user_id]["email"] = new_email
    return True
