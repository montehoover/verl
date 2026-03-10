# Simulating a database
db_store = {
    "users": {
        "1": {"cell": "123-456-7890"},
    }
}

def get_user_phone(request_data):
    """
    Retrieves the phone number of a user from the database.

    Args:
        request_data (dict): A dictionary containing the user ID in its body.
                             Example: {"body": {"user_id": "1"}}

    Returns:
        str: The phone number if the user exists, or an empty string otherwise.
    """
    try:
        user_id = request_data.get("body", {}).get("user_id")
        if user_id:
            user_info = db_store.get("users", {}).get(user_id)
            if user_info and "cell" in user_info:
                return user_info["cell"]
    except (AttributeError, TypeError):
        # Handle cases where request_data or its nested keys are not as expected
        pass
    return ""
