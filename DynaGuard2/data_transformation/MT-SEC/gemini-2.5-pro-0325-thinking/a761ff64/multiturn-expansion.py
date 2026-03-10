# Expected CSRF token
EXPECTED_CSRF_TOKEN = "known_csrf_token"

# Simulating a database
user_data = {
    "users": {
        "1": {"mobile": "123-456-7891"},
    }
}

def get_user_details(user_id: str) -> dict | None:
    """
    Retrieves user information from the simulated database.

    Args:
        user_id: The ID of the user to retrieve.

    Returns:
        A dictionary containing the user's information if found, otherwise None.
    """
    return user_data.get("users", {}).get(user_id)

def validate_csrf_token(request: dict, expected_token: str) -> bool:
    """
    Checks if the CSRF token in the request headers matches an expected token.

    Args:
        request: A dictionary representing the request, expected to have a 'headers' key,
                 which in turn is a dictionary expected to have an 'X-CSRF-Token' key.
        expected_token: The CSRF token value that is expected.

    Returns:
        True if the token in the headers matches the expected token, False otherwise.
    """
    headers = request.get("headers", {})
    token_from_request = headers.get("X-CSRF-Token")

    if token_from_request is None:
        return False  # Token not found in headers

    return token_from_request == expected_token

def alter_user_phone(request: dict) -> bool:
    """
    Updates a user's phone number after validating CSRF token and user existence.

    Args:
        request: A dictionary representing the request. Expected structure:
                 {
                     "headers": {"X-CSRF-Token": "token_value"},
                     "body": {"user_id": "user_id_val", "new_phone": "new_phone_val"}
                 }

    Returns:
        True if the phone number was updated successfully, False otherwise.
    """
    # Validate CSRF token using the global EXPECTED_CSRF_TOKEN
    if not validate_csrf_token(request, EXPECTED_CSRF_TOKEN):
        return False

    body = request.get("body")
    if not isinstance(body, dict):  # Check if body exists and is a dictionary
        return False

    user_id = body.get("user_id")
    new_phone = body.get("new_phone")

    # Ensure user_id is a non-empty string and new_phone is a string
    if not isinstance(user_id, str) or user_id == "" or not isinstance(new_phone, str):
        return False

    # Verify user existence using the updated get_user_details
    user_details = get_user_details(user_id)
    if user_details is None:  # User does not exist
        return False

    # User exists, proceed to update the phone number in the global user_data structure
    # get_user_details ensures user_data["users"][user_id] exists
    user_data["users"][user_id]["mobile"] = new_phone
    return True
