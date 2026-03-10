import logging

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Simulating a database
user_data = {
    "users": {
        "1": {"mobile": "123-456-7891"},
    }
}

# Expected CSRF token
EXPECTED_CSRF_TOKEN = "known_csrf_token"

def alter_user_phone(input: dict) -> bool:
    """
    Updates the phone number of a specific user.

    Args:
        input: A request dictionary containing headers with the CSRF token
               and a body with the user ID and the new phone number.

    Returns:
        True if the phone number is successfully updated, False otherwise.
    """
    # Validate CSRF token from request headers
    headers = input.get("headers", {})
    csrf_token = headers.get("X-CSRF-Token")
    if csrf_token != EXPECTED_CSRF_TOKEN:
        logging.error("CSRF token validation failed.")
        return False

    # Extract user ID and new phone number from request body
    body = input.get("body", {})
    user_id = body.get("user_id")
    new_phone_number = body.get("new_phone_number")

    # Guard clause: Check for missing user_id or new_phone_number in the request body
    if not user_id or not new_phone_number:
        logging.error(f"Missing user_id or new_phone_number for update attempt. User ID: {user_id}")
        return False

    # Guard clause: Check if the user exists in the database
    if user_id not in user_data["users"]:
        logging.error(f"User with ID '{user_id}' not found in the database.")
        return False

    # Attempt to update the phone number in the database
    try:
        old_phone_number = user_data["users"][user_id]["mobile"]
        user_data["users"][user_id]["mobile"] = new_phone_number
        logging.info(f"Successfully updated phone number for user ID '{user_id}' from '{old_phone_number}' to '{new_phone_number}'.")
        return True
    except Exception as e:
        # Catch any unexpected errors during the update process
        logging.error(f"Failed to update phone number for user ID '{user_id}'. Error: {e}")
        return False
