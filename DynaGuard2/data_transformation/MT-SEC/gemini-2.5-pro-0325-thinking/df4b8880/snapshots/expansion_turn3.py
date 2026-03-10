# Simulating a database
record_storage = {
    "users": {
        "1": {"email": "old.email@example.com"},
    }
}

# Expected CSRF token
EXPECTED_CSRF_TOKEN = "secure_csrf_token"

def validate_csrf_token(request_headers: dict) -> bool:
    """
    Validates the CSRF token from request headers.

    Args:
        request_headers: A dictionary representing the request headers.

    Returns:
        True if the CSRF token is valid, False otherwise.
    """
    actual_token = request_headers.get('X-CSRF-Token')
    return actual_token == EXPECTED_CSRF_TOKEN

def get_user_info(user_id: str) -> dict | None:
    """
    Retrieves user information from the record_storage.

    Args:
        user_id: The ID of the user to retrieve.

    Returns:
        A dictionary containing user data if found, otherwise None.
    """
    return record_storage.get("users", {}).get(user_id)

def alter_user_email(request: dict) -> bool:
    """
    Updates a user's email address after validating CSRF token and user existence.

    Args:
        request: A dictionary containing 'headers', 'user_id', and 'new_email'.

    Returns:
        True if the email was updated successfully, False otherwise.
    """
    headers = request.get("headers")
    user_id = request.get("user_id")
    new_email = request.get("new_email")

    if not headers or not user_id or new_email is None: # new_email can be an empty string
        return False

    # Validate CSRF token
    if not validate_csrf_token(headers):
        return False

    # Check if user exists
    user_info = get_user_info(user_id)
    if user_info is None:
        return False

    # Update email
    record_storage["users"][user_id]["email"] = new_email
    return True
