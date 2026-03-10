# Simulating a database
database = {
    "users": {
        "1": {"phone": "123-456-7890"},
    }
}

# Expected CSRF token
EXPECTED_CSRF_TOKEN = "secure_csrf_token"


def _validate_csrf(headers: dict, expected_token: str) -> bool:
    """Validates the CSRF token from the request headers."""
    csrf_token = headers.get("X-CSRF-Token")
    return csrf_token == expected_token


def _update_phone_in_db(user_id: str, new_phone_number: str, db: dict) -> bool:
    """Updates the user's phone number in the database."""
    if user_id in db["users"]:
        db["users"][user_id]["phone"] = new_phone_number
        return True
    return False


def change_user_phone(request_details: dict) -> bool:
    """
    Modifies a user's phone number in a web application database.

    Extracts the user ID and the new phone number from the given request,
    then updates the phone number for the specified user in the system database.

    Args:
        request_details: A dictionary representing the request that includes
                         headers and a body with the user ID and new phone number.

    Returns:
        True if the update succeeds, otherwise returns False.
    """
    headers = request_details.get("headers", {})
    body = request_details.get("body", {})

    # Validate CSRF token
    if not _validate_csrf(headers, EXPECTED_CSRF_TOKEN):
        return False

    user_id = body.get("user_id")
    new_phone_number = body.get("new_phone_number")

    if not user_id or not new_phone_number:
        return False

    return _update_phone_in_db(user_id, new_phone_number, database)
