EXPECTED_CSRF_TOKEN = 'secure_csrf_token'
CSRF_HEADER_NAME = 'X-CSRF-Token'

def validate_request(request_headers: dict) -> bool:
    """
    Validates the request by checking the CSRF token.

    Args:
        request_headers: A dictionary representing the request headers.

    Returns:
        True if the CSRF token is valid, False otherwise.
    """
    actual_token = request_headers.get(CSRF_HEADER_NAME)
    return actual_token == EXPECTED_CSRF_TOKEN

# Simulating a database
database = {
    "users": {
        "1": {"phone": "123-456-7890"},
    }
}

def get_user_info(user_id: str) -> dict | None:
    """
    Retrieves user information from the database.

    Args:
        user_id: The ID of the user to retrieve.

    Returns:
        A dictionary containing user information if the user exists,
        None otherwise.
    """
    return database.get("users", {}).get(user_id)

def alter_user_phone(service_request: dict) -> bool:
    """
    Updates a user's phone number in the database after validating CSRF token
    and user existence.

    Args:
        service_request: A dictionary containing 'headers' and 'body'.
                         The 'headers' should contain the CSRF token.
                         The 'body' should contain 'user_id' and 'new_phone_number'.

    Returns:
        True if the update is successful, False otherwise.
    """
    headers = service_request.get("headers", {})
    if not validate_request(headers):
        return False

    body = service_request.get("body", {})
    user_id = body.get("user_id")
    new_phone_number = body.get("new_phone_number")

    if not user_id or new_phone_number is None: # Check if new_phone_number is None, as empty string might be valid
        return False

    user_info = get_user_info(user_id)
    if user_info is None:
        return False

    # Update the phone number in the database
    # Ensure 'users' key exists and user_id exists within 'users'
    if "users" in database and user_id in database["users"]:
        database["users"][user_id]["phone"] = new_phone_number
        return True
    
    return False
