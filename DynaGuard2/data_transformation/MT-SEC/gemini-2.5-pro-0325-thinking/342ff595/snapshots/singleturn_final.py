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

def remove_user_profile(request_data: dict) -> bool:
    """
    Removes a user's account from the system using their user ID.

    Args:
        request_data: A representation of an HTTP request containing headers
                      and body with the user ID.

    Returns:
        True if the account was removed successfully, otherwise False.
    """
    headers = request_data.get("headers", {})
    body = request_data.get("body", {})

    # Validate CSRF token
    csrf_token = headers.get("X-CSRF-Token")
    if csrf_token != EXPECTED_CSRF_TOKEN:
        return False

    user_id = body.get("user_id")

    if user_id and user_id in database["users"]:
        del database["users"][user_id]
        return True
    
    return False
