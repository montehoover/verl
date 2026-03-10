# Simulating a database
database = {
    "users": {
        "1": {"email": "old.email@example.com"},
    }
}

def check_user_email(request_data: dict) -> bool:
    """
    Checks if the given email matches the user's current email in the database.

    Args:
        request_data: A dictionary containing a body with the user ID and email.
                      Expected structure: {"body": {"user_id": "some_id", "email": "user@example.com"}}

    Returns:
        True if the provided email matches the user's current email, False otherwise.
    """
    try:
        user_id = request_data["body"]["user_id"]
        email_to_check = request_data["body"]["email"]
        
        user_info = database["users"].get(user_id)
        if user_info:
            current_email = user_info.get("email")
            return current_email == email_to_check
        return False
    except (KeyError, TypeError):
        # Handles cases where 'body', 'user_id', or 'email' are missing,
        # or request_data is not a dictionary.
        return False
