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


def validate_csrf_token(headers: dict) -> bool:
    """
    Validates the CSRF token from HTTP headers.

    Args:
        headers: A dictionary representing HTTP headers.

    Returns:
        True if the CSRF token is valid, False otherwise.
    """
    actual_token = headers.get('X-CSRF-Token')
    return actual_token == EXPECTED_CSRF_TOKEN


def get_user_details(user_id: str) -> dict | None:
    """
    Retrieves user details from the simulated database.

    Args:
        user_id: The ID of the user to retrieve.

    Returns:
        A dictionary containing user details if found, otherwise None.
    """
    return database["users"].get(user_id)


def remove_account(request: dict) -> bool:
    """
    Removes a user account after validating the CSRF token.

    Args:
        request: A dictionary containing 'headers' and 'user_id'.
                 'headers' should contain 'X-CSRF-Token'.

    Returns:
        True if the account is successfully deleted, False otherwise.
    """
    headers = request.get("headers")
    user_id = request.get("user_id")

    if not headers or not user_id:
        return False

    if not validate_csrf_token(headers):
        return False

    if user_id in database["users"]:
        del database["users"][user_id]
        return True
    
    return False
