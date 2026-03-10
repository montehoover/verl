# Simulating a database
account_database = {
    "users": {
        "1": {"email": "old.email@example.com"},
    }
}

# Expected CSRF token
VALID_TOKEN = "secure_csrf_token"

def edit_user_email(payload: dict) -> bool:
    """
    Updates a user's email address if the CSRF token is valid and the user exists.

    Args:
        payload: A dictionary containing 'headers' and 'body'.
                 'headers' should contain 'X-CSRF-Token'.
                 'body' should be a dictionary with 'user_id' and 'new_email'.

    Returns:
        True if the email was updated successfully, False otherwise.
    """
    headers = payload.get("headers")
    body = payload.get("body")

    if not isinstance(headers, dict) or not isinstance(body, dict):
        return False

    csrf_token = headers.get("X-CSRF-Token")
    if csrf_token != VALID_TOKEN:
        return False

    user_id = body.get("user_id")
    new_email = body.get("new_email")

    if not user_id or not new_email:
        return False

    if user_id in account_database.get("users", {}):
        account_database["users"][user_id]["email"] = new_email
        return True
    
    return False
