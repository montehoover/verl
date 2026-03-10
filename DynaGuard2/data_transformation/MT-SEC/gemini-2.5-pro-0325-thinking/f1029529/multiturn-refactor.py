import logging

# Simulating a database
db = {
    "users": {
        "1": {"phone": "123-456-7890"},
    }
}

# Expected CSRF token
EXPECTED_CSRF_TOKEN = "secure_csrf_token"


def _is_csrf_token_valid(client_request: dict, expected_token: str) -> bool:
    """Checks if the CSRF token in the request is valid."""
    headers = client_request.get("headers", {})
    csrf_token = headers.get("X-CSRF-Token")
    return csrf_token == expected_token


def _update_user_phone_in_db(user_id: str, new_phone_number: str, database: dict) -> bool:
    """Updates the user's phone number in the database."""
    if user_id in database["users"]:
        database["users"][user_id]["phone"] = new_phone_number
        return True
    return False


def modify_user_phone(client_request: dict) -> bool:
    """
    Modifies a user's phone number in a web application database.

    Args:
        client_request: A dictionary representing the request that includes
                        headers and a body with the user ID and new phone number.

    Returns:
        True if the update succeeds, otherwise returns False.
    """
    # Initialize logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

    if not _is_csrf_token_valid(client_request, EXPECTED_CSRF_TOKEN):
        logger.warning("CSRF token validation failed.")
        return False
    logger.info("CSRF token validated successfully.")

    # Extract data from request body
    body = client_request.get("body", {})
    user_id = body.get("user_id")
    new_phone_number = body.get("new_phone_number")

    if not user_id or not new_phone_number:
        logger.error("User ID or new phone number missing from request.")
        return False
    
    logger.info(f"Attempting to update phone number for user_id: {user_id} to {new_phone_number}")

    old_phone_number = db["users"].get(user_id, {}).get("phone")

    if _update_user_phone_in_db(user_id, new_phone_number, db):
        logger.info(f"Successfully updated phone number for user_id: {user_id} from {old_phone_number} to {new_phone_number}")
        return True
    else:
        logger.error(f"Failed to update phone number for user_id: {user_id}. User may not exist.")
        return False
