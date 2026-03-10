# Simulating a database
database = {
    "users": {
        "1": {"phone": "123-456-7890"},
    }
}

def check_user_exists(user_id: str) -> bool:
    """
    Checks if a user exists in the simulated database.

    Args:
        user_id: The ID of the user to check.

    Returns:
        True if the user exists, False otherwise.
    """
    return user_id in database.get("users", {})

EXPECTED_CSRF_TOKEN = "secure_csrf_token"

def validate_csrf_token(headers: dict) -> bool:
    """
    Validates the CSRF token from the request headers.

    Args:
        headers: A dictionary representing the request headers.

    Returns:
        True if the CSRF token is valid, False otherwise.
    """
    token = headers.get("X-CSRF-Token")
    return token == EXPECTED_CSRF_TOKEN
