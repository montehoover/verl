# Simulating a database
account_database = {
    "users": {
        "1": {"email": "old.email@example.com"},
    }
}


def check_email_match(payload):
    """
    Check whether the provided email matches the user's email in account_database.

    Args:
        payload (dict): A dictionary containing a 'body' field with both the user ID and the email to verify.
                        The 'body' must be a dict and may contain:
                          - user id in one of: 'user_id', 'id', 'userId', 'userID'
                          - email in one of: 'email', 'user_email', 'emailAddress', 'userEmail'

    Returns:
        bool: True if the user exists and the email matches, otherwise False.
    """
    if not isinstance(payload, dict):
        return False

    body = payload.get("body")
    if not isinstance(body, dict):
        return False

    # Extract user_id from body
    user_id = None
    for key in ("user_id", "id", "userId", "userID"):
        if key in body and body[key] is not None:
            user_id = str(body[key]).strip()
            break

    # Extract email from body
    provided_email = None
    for key in ("email", "user_email", "emailAddress", "userEmail"):
        if key in body and body[key] is not None:
            provided_email = str(body[key]).strip()
            break

    if not user_id or not provided_email:
        return False

    users = account_database.get("users", {})
    user_record = users.get(user_id)
    if not isinstance(user_record, dict):
        return False

    stored_email = user_record.get("email")
    if not isinstance(stored_email, str):
        return False

    # Normalize emails for comparison
    return stored_email.strip().lower() == provided_email.strip().lower()
