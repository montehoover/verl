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

def is_csrf_token_valid(request: dict) -> bool:
    """
    Validates the CSRF token from the request headers.

    Args:
        request: dict, The request object containing headers.

    Returns:
        bool, True if the CSRF token is valid, False otherwise.
    """
    if not isinstance(request, dict):
        return False
    headers = request.get("headers")
    if not isinstance(headers, dict):
        return False
    token = headers.get("X-CSRF-Token")
    return token == EXPECTED_CSRF_TOKEN

def _delete_user_from_db(user_id) -> bool:
    """
    Deletes a user from the database.

    Args:
        user_id: The ID of the user to delete.

    Returns:
        bool, True if the user was found and deleted, False otherwise.
    """
    if user_id in database["users"]:
        del database["users"][user_id]
        return True
    return False

def delete_user_account(request: dict) -> bool:
    """
    Deletes a user account after validating CSRF token and user existence.

    Args:
        request: dict, The request object containing headers and body with user ID.

    Returns:
        bool, True if the user is deleted successfully, False otherwise.
    """
    if not isinstance(request, dict):
        return False

    if not is_csrf_token_valid(request):
        return False
    
    body = request.get("body")
    if not isinstance(body, dict):
        return False

    user_id = body.get("user_id")
    if user_id is None: # Allow user_id to be any type that can be a dict key
        return False

    return _delete_user_from_db(user_id)
