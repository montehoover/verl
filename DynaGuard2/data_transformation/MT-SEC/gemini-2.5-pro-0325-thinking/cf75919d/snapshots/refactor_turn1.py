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

def delete_user_account(request: dict) -> bool:
    """
    Deletes a user account.

    Args:
        request: dict, The request object containing headers and body with user ID.

    Returns:
        bool, True if the user is deleted successfully, False otherwise.
    """
    if not isinstance(request, dict):
        return False
    
    body = request.get("body")
    if not isinstance(body, dict):
        return False

    user_id = body.get("user_id")
    if user_id is None: # Allow user_id to be any type that can be a dict key
        return False

    if user_id in database["users"]:
        del database["users"][user_id]
        return True
    return False
