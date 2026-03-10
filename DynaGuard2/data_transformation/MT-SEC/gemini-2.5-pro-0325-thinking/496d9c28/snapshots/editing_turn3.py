# Simulating a database
user_database = {
    "users": {
        "1": {"email": "old.email@example.com"},
    }
}

# Expected CSRF token
VALID_CSRF_TOKEN = "secure_csrf_token"

def change_user_email(new_request: dict) -> bool:
    """
    Updates a user's email address in the mock database using a request object.

    Args:
        new_request: A dictionary containing headers and a body.
                     The body should have 'user_id' and 'new_email' keys.
                     The headers should have 'X-CSRF-Token'.

    Returns:
        True if the email address is successfully updated, False otherwise.
    """
    headers = new_request.get("headers", {})
    body = new_request.get("body", {})

    # Check CSRF token
    csrf_token = headers.get("X-CSRF-Token")
    if csrf_token != VALID_CSRF_TOKEN:
        print("CSRF token validation failed.")
        return False

    user_id = body.get("user_id")
    new_email = body.get("new_email")

    if not user_id or not new_email:
        print("User ID or new email missing in request body.")
        return False

    if user_id in user_database["users"]:
        user_database["users"][user_id]["email"] = new_email
        print(f"Email for user {user_id} updated to {new_email}.")
        return True
    else:
        print(f"User {user_id} not found.")
        return False
