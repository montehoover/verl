import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("phone_changes.log"),
        logging.StreamHandler()  # Optional: To also see logs in console
    ]
)

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


def _update_phone_in_db(user_id: str, new_phone_number: str, db: dict) -> tuple[bool, str | None]:
    """
    Updates the user's phone number in the database.
    Returns a tuple: (success_status, old_phone_number).
    old_phone_number is None if user_id not found.
    """
    if user_id in db["users"]:
        old_phone_number = db["users"][user_id].get("phone")
        db["users"][user_id]["phone"] = new_phone_number
        return True, old_phone_number
    return False, None


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

    user_id = body.get("user_id")
    new_phone_number = body.get("new_phone_number")

    logging.info(f"Attempting phone number change for user_id: {user_id} to new_phone_number: {new_phone_number}")

    # Validate CSRF token
    if not _validate_csrf(headers, EXPECTED_CSRF_TOKEN):
        logging.warning(f"CSRF token validation failed for user_id: {user_id}.")
        return False

    if not user_id or not new_phone_number:
        logging.warning(f"Missing user_id or new_phone_number in request for user_id: {user_id}.")
        return False

    success, old_phone_number = _update_phone_in_db(user_id, new_phone_number, database)

    if success:
        logging.info(
            f"Successfully updated phone for user_id: {user_id}. "
            f"Old phone: {old_phone_number}, New phone: {new_phone_number}."
        )
    else:
        if old_phone_number is None: # Implies user was not found
            logging.error(
                f"Failed to update phone for user_id: {user_id}. User not found."
            )
        else: # Should not happen with current _update_phone_in_db logic if user_id was found but update failed for other reasons
             logging.error(
                f"Failed to update phone for user_id: {user_id}. "
                f"Attempted to change from {old_phone_number} to {new_phone_number}."
            )
    return success
