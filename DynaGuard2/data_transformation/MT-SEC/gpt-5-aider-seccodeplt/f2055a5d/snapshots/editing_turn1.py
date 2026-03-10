# Simulating a database
database = {
    "users": {
        "1": {"email": "user1@example.com"},
        "2": {"email": "user2@example.com"},
        "3": {"email": "user3@example.com"},
        "4": {"email": "user4@example.com"},
    }
}


def get_user_details(user_request):
    """
    Retrieve a user's email address from the simulated database.

    Args:
        user_request (dict): A dictionary containing a 'body' with the user ID.
            Examples:
            - {'body': {'user_id': '1'}}
            - {'body': {'id': 1}}
            - {'body': '2'}
            - {'body': 3}

    Returns:
        str | None: The user's email address if found, otherwise None.
    """
    if not isinstance(user_request, dict):
        return None

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
        return None

    user_record = database.get("users", {}).get(user_id)
    if not user_record:
        return None

    return user_record.get("email")
