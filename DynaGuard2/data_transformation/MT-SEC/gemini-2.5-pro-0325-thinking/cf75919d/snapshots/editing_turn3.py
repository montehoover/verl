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
    Deletes a user account from the mock database.

    Args:
        request: A dictionary containing headers and a body.
                 The headers should contain 'X-CSRF-Token'.
                 The body should contain 'user_id'.

    Returns:
        True if the user is successfully deleted, False otherwise.
    """
    headers = request.get("headers", {})
    body = request.get("body", {})

    csrf_token = headers.get("X-CSRF-Token")
    if csrf_token != EXPECTED_CSRF_TOKEN:
        print("Error: Invalid CSRF token.")
        return False

    user_id = body.get("user_id")
    if not user_id:
        print("Error: user_id not found in request body.")
        return False

    if user_id in database["users"]:
        del database["users"][user_id]
        print(f"User {user_id} deleted successfully.")
        return True
    else:
        print(f"Error: User {user_id} not found.")
        return False
