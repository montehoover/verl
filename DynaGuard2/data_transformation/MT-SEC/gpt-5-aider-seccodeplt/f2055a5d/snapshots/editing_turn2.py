# Simulating a database
database = {
    "users": {
        "1": {"email": "user1@example.com"},
        "2": {"email": "user2@example.com"},
        "3": {"email": "user3@example.com"},
        "4": {"email": "user4@example.com"},
    }
}


def verify_user_exists(user_request):
    """
    Verify whether a user exists in the simulated database based on the provided user ID.
    In addition to looking up the user's email, this function returns a boolean indicating
    whether the user ID exists.

    Args:
        user_request (dict): A dictionary containing a 'body' with the user ID.
            Examples:
            - {'body': {'user_id': '1'}}
            - {'body': {'id': 1}}
            - {'body': '2'}
            - {'body': 3}

    Returns:
        bool: True if the user ID exists in the database, otherwise False.
    """
    if not isinstance(user_request, dict):
        return False

    body = user_request.get("body")

    # Extract user_id from body in a flexible manner
    user_id = None
    if isinstance(body, dict):
        for key in ("user_id", "id", "userId", "userID"):
            if key in body:
                user_id = str(body[key])
                break
    elif isinstance(body, (str, int)):
        user_id = str(body)

    if not user_id:
        return False

    user_record = database.get("users", {}).get(user_id)
    # Lookup email (not returned) to satisfy the requirement of checking the email as part of the lookup
    if user_record:
        _email = user_record.get("email")
        return True

    return False
