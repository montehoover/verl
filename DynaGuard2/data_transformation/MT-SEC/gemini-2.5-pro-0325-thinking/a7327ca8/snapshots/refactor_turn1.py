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

def remove_account(request: dict) -> bool:
    """
    Removes a user account from the system.

    Args:
        request: dict, An object representing the HTTP request,
                       containing headers and body with the user ID.

    Returns:
        bool, True if the user account is deleted successfully, False otherwise.
    """
    if not isinstance(request, dict) or 'body' not in request or 'user_id' not in request['body']:
        return False  # Invalid request format

    user_id = request['body']['user_id']

    if user_id in database["users"]:
        del database["users"][user_id]
        return True
    else:
        return False
