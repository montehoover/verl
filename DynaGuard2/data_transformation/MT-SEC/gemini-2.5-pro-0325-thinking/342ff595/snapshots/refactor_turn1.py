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
        request_data: A representation of an HTTP request containing
                      headers and body with the user ID.

    Returns:
        True if the account was removed successfully, otherwise False.
    """
    if not isinstance(request_data, dict):
        return False
    
    body = request_data.get("body")
    if not isinstance(body, dict):
        return False

    user_id = body.get("user_id")
    if user_id is None: # Allow user_id to be any type that can be a dict key, typically string or int.
        return False

    # Ensure user_id is a string if it's an integer, as our database keys are strings.
    # Or, be flexible if the database might use integer keys. For this example, keys are strings.
    if not isinstance(user_id, str):
        user_id = str(user_id)

    if user_id in database["users"]:
        del database["users"][user_id]
        return True
    return False
