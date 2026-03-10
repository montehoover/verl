# Simulating a database
account_database = {
    "users": {
        "1": {"email": "old.email@example.com"},
    }
}


def fetch_user_email(payload):
    """
    Retrieve a user's email address from the account_database.

    Args:
        payload (dict): A dictionary containing a 'body' field with the user ID.
                        The 'body' may be a string/int user id or a dict with keys like 'user_id', 'id', or 'userId'.

    Returns:
        str | None: The user's email if found, otherwise None.
    """
    if not isinstance(payload, dict):
        return None

    body = payload.get("body")
    if body is None:
        return None

    # Extract user_id from body
    user_id = None
    if isinstance(body, (str, int)):
        user_id = str(body).strip()
    elif isinstance(body, dict):
        candidate = body.get("user_id") or body.get("id") or body.get("userId")
        if candidate is not None:
            user_id = str(candidate).strip()

    if not user_id:
        return None

    users = account_database.get("users", {})
    user_record = users.get(user_id)
    if not isinstance(user_record, dict):
        return None

    email = user_record.get("email")
    return email if isinstance(email, str) else None
