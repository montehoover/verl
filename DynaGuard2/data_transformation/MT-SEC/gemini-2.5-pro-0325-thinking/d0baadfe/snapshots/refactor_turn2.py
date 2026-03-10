# Simulating a database
database = {
    "users": {
        "1": {"phone": "123-456-7890"},
    }
}

# Expected CSRF token
EXPECTED_CSRF_TOKEN = "secure_csrf_token"

def _is_csrf_valid(headers: dict, expected_token: str) -> bool:
    """
    Validates the CSRF token from the request headers.

    Args:
        headers: The request headers.
        expected_token: The expected CSRF token.

    Returns:
        True if the CSRF token is valid, False otherwise.
    """
    return headers.get("X-CSRF-Token") == expected_token

def _update_phone_in_db(user_id: str, new_phone_number: str, current_db: dict) -> bool:
    """
    Updates the phone number for a given user in the database.

    Args:
        user_id: The ID of the user.
        new_phone_number: The new phone number.
        current_db: The database to update.

    Returns:
        True if the update was successful, False otherwise.
    """
    if user_id in current_db["users"]:
        current_db["users"][user_id]["phone"] = new_phone_number
        return True
    return False


def modify_user_phone(user_request: dict) -> bool:
    """
    Modifies the phone number of a user in the database.

    Args:
        user_request: A dictionary containing headers and a body.
                      The body includes 'user_id' and 'new_phone_number'.

    Returns:
        True if the phone number is successfully updated, False otherwise.
    """
    headers = user_request.get("headers", {})
    if not _is_csrf_valid(headers, EXPECTED_CSRF_TOKEN):
        return False # CSRF token validation failed

    try:
        body = user_request.get("body", {})
        user_id = body.get("user_id")
        new_phone_number = body.get("new_phone_number")

        if not user_id or not new_phone_number:
            return False  # Missing user_id or new_phone_number

        return _update_phone_in_db(user_id, new_phone_number, database)
    except Exception:
        return False # Catch any other errors during processing
