# Expected CSRF token
EXPECTED_CSRF_TOKEN = "secure_csrf_token"

def validate_request(headers: dict) -> bool:
    """
    Checks if the request's headers contain a valid CSRF token.

    Args:
        headers: A dictionary representing HTTP headers.

    Returns:
        True if the CSRF token is valid, False otherwise.
    """
    token = headers.get("X-CSRF-Token")
    return token == EXPECTED_CSRF_TOKEN

# Simulating a database
database = {
    "users": {
        "1": {"email": "user1@example.com"},
        "2": {"email": "user2@example.com"},
        "3": {"email": "user3@example.com"},
        "4": {"email": "user4@example.com"},
    }
}

def check_user_existence(user_id: str) -> bool:
    """
    Verifies whether a user with a given ID exists in the system's database.

    Args:
        user_id: The ID of the user to check.

    Returns:
        True if the user is found, False if not.
    """
    return user_id in database["users"]

def remove_user_account(input_request: dict) -> bool:
    """
    Removes a user account after validating CSRF token and checking user existence.

    Args:
        input_request: A dictionary containing 'headers' for CSRF validation
                       and 'user_id' for user identification.

    Returns:
        True if the user was successfully removed, False otherwise.
    """
    if not validate_request(input_request.get("headers", {})):
        return False

    user_id = input_request.get("user_id")
    if not user_id:
        return False

    if check_user_existence(user_id):
        if user_id in database["users"]:
            del database["users"][user_id]
            return True
    return False
