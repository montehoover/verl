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

def terminate_user_account(request_data: dict) -> bool:
    """
    Facilitates the deletion of a user's account within the application.

    Args:
        request_data: A dictionary representing an HTTP request,
                      with 'headers' for CSRF token and 'body'
                      containing user ID.

    Returns:
        True if the user's account is successfully deleted; otherwise, False.
    """
    # Validate CSRF token
    csrf_token = request_data.get("headers", {}).get("X-CSRF-Token")
    if csrf_token != EXPECTED_CSRF_TOKEN:
        return False

    # Get user ID from request body
    user_id = request_data.get("body", {}).get("user_id")
    if not user_id:
        return False

    # Check if user exists and delete
    if user_id in database["users"]:
        del database["users"][user_id]
        return True
    
    return False
