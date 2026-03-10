# Simulating a database
database = {
    "users": {
        "1": {"phone": "123-456-7890"},
    }
}

# Expected CSRF token
EXPECTED_CSRF_TOKEN = "secure_csrf_token"


def _is_csrf_valid(transaction_request: dict, expected_token: str) -> bool:
    """Validates the CSRF token from the request headers."""
    csrf_token = transaction_request.get("headers", {}).get("X-CSRF-Token")
    return csrf_token == expected_token


def _extract_user_data_from_request(request_body: dict) -> tuple[str | None, str | None]:
    """Extracts user ID and new phone number from the request body."""
    user_id = request_body.get("user_id")
    new_phone_number = request_body.get("new_phone_number")
    return user_id, new_phone_number


def _update_user_phone_in_db(user_id: str, new_phone_number: str, db: dict) -> bool:
    """Updates the user's phone number in the database."""
    if user_id in db["users"]:
        db["users"][user_id]["phone"] = new_phone_number
        return True
    return False


def change_user_contact(transaction_request: dict) -> bool:
    """
    Modifies the phone number of a user in the application's database.

    Args:
        transaction_request: A dictionary that contains headers and a body;
                             the body includes the user ID and the new phone number.

    Returns:
        True if the phone number is successfully updated, False otherwise.
    """
    try:
        if not _is_csrf_valid(transaction_request, EXPECTED_CSRF_TOKEN):
            return False  # CSRF token validation failed

        body = transaction_request.get("body", {})
        user_id, new_phone_number = _extract_user_data_from_request(body)

        if not user_id or not new_phone_number:
            return False  # Missing user_id or new_phone_number

        if not _update_user_phone_in_db(user_id, new_phone_number, database):
            return False  # User not found or update failed

        return True
    except Exception:
        # Log the exception in a real application
        return False
