import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Simulating a database
db_users = {
    "users": {
        "1": {"telephone": "123-556-7890"},
    }
}

# Expected CSRF token
EXPECTED_CSRF_TOKEN = "csrf_token_secured"

def change_user_phone(payload: dict) -> bool:
    """
    Updates the phone number of a specific user.

    Args:
        payload: A request dictionary containing headers with the CSRF token
                 and a body with the user ID and the new phone number.

    Returns:
        True if the phone number is successfully updated, False otherwise.
    """
    # Validate CSRF token
    csrf_token = payload.get("headers", {}).get("X-CSRF-Token")
    if csrf_token != EXPECTED_CSRF_TOKEN:
        logging.warning("CSRF token validation failed.")
        return False

    # Get user ID and new phone number from payload body
    body = payload.get("body", {})
    user_id = body.get("user_id")
    new_phone_number = body.get("phone_number")

    if not user_id:
        logging.error("User ID not provided in payload.")
        return False
    if not new_phone_number:
        logging.error(f"New phone number not provided for user ID: {user_id}.")
        return False

    logging.info(f"Attempting to update phone number for user ID: {user_id}.")

    # Update phone number in the database
    if user_id not in db_users["users"]:
        logging.error(f"User ID: {user_id} not found in database.")
        # User not found
        return False

    db_users["users"][user_id]["telephone"] = new_phone_number
    logging.info(f"Successfully updated phone number for user ID: {user_id}.")
    return True
