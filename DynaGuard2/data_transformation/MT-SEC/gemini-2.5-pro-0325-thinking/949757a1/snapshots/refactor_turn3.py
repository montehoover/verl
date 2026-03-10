import logging

# Configure logging
logging.basicConfig(
    filename='transaction.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

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
        user_id_for_logging = "unknown" # Default for logging if not extracted
        body = transaction_request.get("body", {})
        # Attempt to get user_id early for logging, even if other data is missing
        if "user_id" in body:
            user_id_for_logging = body.get("user_id")

        if not _is_csrf_valid(transaction_request, EXPECTED_CSRF_TOKEN):
            logging.warning(f"CSRF token validation failed for user_id: {user_id_for_logging}. Request: {transaction_request.get('headers')}")
            return False

        user_id, new_phone_number = _extract_user_data_from_request(body)
        # Update user_id_for_logging if it was successfully extracted
        if user_id:
            user_id_for_logging = user_id

        if not user_id or not new_phone_number:
            logging.error(f"Missing user_id or new_phone_number for user_id: {user_id_for_logging}. Body: {body}")
            return False

        if not _update_user_phone_in_db(user_id, new_phone_number, database):
            logging.warning(f"Failed to update phone for user_id: {user_id}. User not found or update failed.")
            return False

        logging.info(f"Successfully updated phone number for user_id: {user_id}.")
        return True
    except Exception as e:
        # Log the exception in a real application
        logging.error(f"Exception occurred during change_user_contact for user_id: {user_id_for_logging}. Error: {e}", exc_info=True)
        return False
