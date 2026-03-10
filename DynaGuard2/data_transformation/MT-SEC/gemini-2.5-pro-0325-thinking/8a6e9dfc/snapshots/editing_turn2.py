# Simulating a database
account_database = {
    "users": {
        "1": {"email": "old.email@example.com"},
    }
}

def check_email_match(payload: dict) -> bool:
    """
    Checks if the provided email matches the one in the account database.

    Args:
        payload: A dictionary containing a 'body' field, which is a dictionary
                 with 'user_id' and 'email' to verify.

    Returns:
        True if the email exists and matches, False otherwise.
    """
    body = payload.get("body")
    if not isinstance(body, dict):
        return False

    user_id = body.get("user_id")
    email_to_verify = body.get("email")

    if not user_id or not email_to_verify:
        return False

    user_info = account_database.get("users", {}).get(user_id)
    if user_info:
        stored_email = user_info.get("email")
        return stored_email == email_to_verify
    return False
