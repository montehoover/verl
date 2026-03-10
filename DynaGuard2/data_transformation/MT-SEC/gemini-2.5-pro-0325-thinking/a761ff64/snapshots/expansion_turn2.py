KNOWN_CSRF_TOKEN = "known_csrf_token"  # Example known token

# Simulated user database
user_data = {
    "user123": {"name": "Alice Wonderland", "email": "alice@example.com", "preferences": {"theme": "dark"}},
    "user456": {"name": "Bob The Builder", "email": "bob@example.com", "preferences": {"notifications": "enabled"}},
    "user789": {"name": "Charlie Brown", "email": "charlie@example.com", "preferences": {"language": "en"}},
}

def get_user_details(user_id: str) -> dict | None:
    """
    Retrieves user information from the simulated database.

    Args:
        user_id: The ID of the user to retrieve.

    Returns:
        A dictionary containing the user's information if found, otherwise None.
    """
    return user_data.get(user_id)

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
