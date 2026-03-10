# Simulating a database
account_database = {
    "users": {
        "1": {"email": "old.email@example.com"},
    }
}

def fetch_user_email(payload: dict) -> str | None:
    """
    Retrieves a user's email address from the account database.

    Args:
        payload: A dictionary containing a 'body' field with the user ID.

    Returns:
        The email address as a string if the user exists, otherwise None.
    """
    user_id = payload.get("body")
    if user_id:
        user_info = account_database.get("users", {}).get(user_id)
        if user_info:
            return user_info.get("email")
    return None
