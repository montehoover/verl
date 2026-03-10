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

# Simulating a database
db = {
    "users": {
        "1": {"phone": "123-456-7890"},
    }
}

# Expected CSRF token
EXPECTED_CSRF_TOKEN = "secure_csrf_token"

def get_user_info(user_id: str) -> dict | None:
    """
    Retrieves user information from the database.

    Args:
        user_id: The ID of the user to retrieve.

    Returns:
        A dictionary containing user details if found, otherwise None.
    """
    return db.get("users", {}).get(user_id)

def modify_user_phone(client_request: dict) -> bool:
    """
    Updates a user's phone number securely.

    Args:
        client_request: A dictionary containing 'headers' and 'body'.
                        The 'body' should contain 'user_id' and 'new_phone_number'.

    Returns:
        True if the update is successful, False otherwise.
    """
    headers = client_request.get("headers", {})
    body = client_request.get("body", {})

    if not validate_csrf_token(headers):
        return False

    user_id = body.get("user_id")
    new_phone_number = body.get("new_phone_number")

    if not user_id or not new_phone_number:
        return False

    user_info = get_user_info(user_id)
    if not user_info:
        return False

    # Update phone number in the database
    db["users"][user_id]["phone"] = new_phone_number
    return True
