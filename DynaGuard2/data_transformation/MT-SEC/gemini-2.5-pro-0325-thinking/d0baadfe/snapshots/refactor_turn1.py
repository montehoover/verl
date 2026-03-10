# Simulating a database
database = {
    "users": {
        "1": {"phone": "123-456-7890"},
    }
}

# Expected CSRF token
EXPECTED_CSRF_TOKEN = "secure_csrf_token"

def modify_user_phone(user_request: dict) -> bool:
    """
    Modifies the phone number of a user in the database.

    Args:
        user_request: A dictionary containing headers and a body.
                      The body includes 'user_id' and 'new_phone_number'.

    Returns:
        True if the phone number is successfully updated, False otherwise.
    """
    try:
        body = user_request.get("body", {})
        user_id = body.get("user_id")
        new_phone_number = body.get("new_phone_number")

        if not user_id or not new_phone_number:
            return False  # Missing user_id or new_phone_number

        if user_id in database["users"]:
            database["users"][user_id]["phone"] = new_phone_number
            return True
        else:
            return False  # User not found
    except Exception:
        return False # Catch any other errors during processing
