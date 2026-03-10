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

def _is_csrf_token_valid(request: dict) -> bool:
    """
    Validates the CSRF token from the request headers.

    Args:
        request: dict, An object representing the HTTP request.

    Returns:
        bool, True if the CSRF token is valid, False otherwise.
    """
    if not isinstance(request, dict) or 'headers' not in request:
        return False
    token = request['headers'].get('X-CSRF-Token')
    return token == EXPECTED_CSRF_TOKEN

def _delete_user(user_id: str) -> bool:
    """
    Deletes a user from the database.

    Args:
        user_id: str, The ID of the user to delete.

    Returns:
        bool, True if the user was deleted successfully, False otherwise.
    """
    if user_id in database["users"]:
        del database["users"][user_id]
        return True
    return False

def remove_account(request: dict) -> bool:
    """
    Removes a user account from the system.

    Args:
        request: dict, An object representing the HTTP request,
                       containing headers and body with the user ID.

    Returns:
        bool, True if the user account is deleted successfully, False otherwise.
    """
    if not isinstance(request, dict) or \
       'body' not in request or \
       'user_id' not in request['body']:
        return False  # Invalid request format for user_id

    if not _is_csrf_token_valid(request):
        return False  # CSRF token invalid or missing

    user_id = request['body']['user_id']

    return _delete_user(user_id)
