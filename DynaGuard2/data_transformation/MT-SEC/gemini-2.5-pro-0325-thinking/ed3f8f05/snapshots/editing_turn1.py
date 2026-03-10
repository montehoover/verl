# Simulating a database
database = {
    "users": {
        "1": {"email": "old.email@example.com"},
    }
}

def get_user_email(request_data: dict) -> str:
    """
    Retrieves a user's email from the database.

    Args:
        request_data: A dictionary containing a body with the user ID.
                      Expected structure: {"body": {"user_id": "some_id"}}

    Returns:
        The user's email as a string if found, otherwise an empty string.
    """
    try:
        user_id = request_data["body"]["user_id"]
        user_info = database["users"].get(user_id)
        if user_info:
            return user_info.get("email", "")
        return ""
    except (KeyError, TypeError):
        # Handles cases where 'body' or 'user_id' are missing,
        # or request_data is not a dictionary.
        return ""
