# Simulating a database
db_store = {
    "users": {
        "1": {"cell": "123-456-7890"},
    }
}

# Expected CSRF token
EXPECTED_CSRF_TOKEN = "secure_csrf_token_value"

def _is_request_valid(request_data: dict, expected_token: str) -> bool:
    """
    Validates the request data including CSRF token and required fields.

    Args:
        request_data: The request dictionary.
        expected_token: The expected CSRF token.

    Returns:
        True if the request is valid, False otherwise.
    """
    # Validate CSRF token
    csrf_token = request_data.get("headers", {}).get("X-CSRF-Token")
    if csrf_token != expected_token:
        return False

    # Get user ID and new phone number from request body
    body = request_data.get("body", {})
    user_id = body.get("user_id")
    new_phone_number = body.get("new_phone_number")

    if not user_id or not new_phone_number:
        return False
    return True

def _update_user_phone_in_db(user_id: str, new_phone_number: str, database: dict) -> bool:
    """
    Updates the user's phone number in the provided database.

    Args:
        user_id: The ID of the user to update.
        new_phone_number: The new phone number.
        database: The database store.

    Returns:
        True if the update was successful, False otherwise.
    """
    if user_id in database["users"]:
        database["users"][user_id]["cell"] = new_phone_number
        return True
    else:
        return False

def modify_user_phone(request_data: dict) -> bool:
    """
    Updates the phone number of a specific user.

    Args:
        request_data: A request dictionary containing headers with the CSRF token
                      and a body with the user ID and the new phone number.

    Returns:
        True if the phone number is successfully updated, False if the update fails.
    """
    if not _is_request_valid(request_data, EXPECTED_CSRF_TOKEN):
        return False

    # Extract data after validation (we know body, user_id, new_phone_number exist)
    body = request_data.get("body", {}) 
    user_id = body.get("user_id")
    new_phone_number = body.get("new_phone_number")

    return _update_user_phone_in_db(user_id, new_phone_number, db_store)
