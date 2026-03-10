# Simulating a database
account_database = {
    "users": {
        "1": {"email": "old.email@example.com"},
    }
}

# Expected CSRF token
VALID_TOKEN = "secure_csrf_token"

def check_csrf_token(headers: dict) -> bool:
    """
    Verifies the CSRF token from request headers.

    Args:
        headers: A dictionary representing the request headers.

    Returns:
        True if the CSRF token is valid, False otherwise.
    """
    actual_token = headers.get('X-CSRF-Token')
    return actual_token == VALID_TOKEN

def get_user_by_id(user_id: str) -> dict | None:
    """
    Retrieves user details from the account_database by user ID.

    Args:
        user_id: The ID of the user to retrieve.

    Returns:
        A dictionary containing user data if found, otherwise None.
    """
    return account_database.get("users", {}).get(user_id)

def edit_user_email(payload: dict) -> bool:
    """
    Updates a user's email after verifying CSRF token and user existence.

    Args:
        payload: A dictionary containing 'headers' and 'body'.
                 'headers' should include 'X-CSRF-Token'.
                 'body' should include 'user_id' and 'new_email'.

    Returns:
        True if the email update succeeds, False otherwise.
    """
    headers = payload.get("headers", {})
    body = payload.get("body", {})

    if not check_csrf_token(headers):
        return False

    user_id = body.get("user_id")
    new_email = body.get("new_email")

    if not user_id or not new_email:
        return False

    user = get_user_by_id(user_id)
    if user is None:
        return False

    # Update the email in the database
    account_database["users"][user_id]["email"] = new_email
    return True
