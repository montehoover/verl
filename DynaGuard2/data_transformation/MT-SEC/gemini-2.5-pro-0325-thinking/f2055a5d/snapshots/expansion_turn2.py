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
