import logging

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Simulating a database
database = {
    "users": {
        "1": {"email": "user1@example.com"},
        "2": {"email": "user2@example.com"},
        "3": {"email": "user3@example.com"},
        "4": {"email": "user4@example.com"},
    }
}

# Expected CSRF token
EXPECTED_CSRF_TOKEN = "secure_csrf_token"

def _is_csrf_token_valid(request_data: dict, expected_token: str) -> bool:
    """
    Validates the CSRF token from the request headers.

    Args:
        request_data: The request data dictionary.
        expected_token: The expected CSRF token.

    Returns:
        True if the token is valid, False otherwise.
    """
    headers = request_data.get("headers")
    if not isinstance(headers, dict):
        return False
    token = headers.get("X-CSRF-Token")
    return token == expected_token

def _delete_user_from_db(user_id: str, db: dict) -> bool:
    """
    Deletes a user from the database.

    Args:
        user_id: The ID of the user to delete.
        db: The database dictionary.

    Returns:
        True if the user was deleted, False otherwise.
    """
    if user_id in db["users"]:
        del db["users"][user_id]
        return True
    return False

def remove_user_profile(request_data: dict) -> bool:
    """
    Removes a user's account from the system using their user ID,
    after validating the CSRF token.

    Args:
        request_data: A representation of an HTTP request containing
                      headers and body with the user ID.

    Returns:
        True if the account was removed successfully, otherwise False.
    """
    if not isinstance(request_data, dict):
        logging.error("Invalid request_data: not a dictionary.")
        return False

    body = request_data.get("body")
    if not isinstance(body, dict):
        logging.error("Invalid request_data: 'body' is not a dictionary or is missing.")
        return False

    user_id_original = body.get("user_id")
    if user_id_original is None:
        logging.error("Invalid request_data: 'user_id' is missing from 'body'.")
        return False

    # Ensure user_id is a string for database lookup and logging
    user_id = str(user_id_original)

    logging.info(f"Processing remove_user_profile request for user_id: {user_id}")

    if not _is_csrf_token_valid(request_data, EXPECTED_CSRF_TOKEN):
        logging.warning(f"CSRF token validation failed for user_id: {user_id}")
        return False
    logging.info(f"CSRF token validation successful for user_id: {user_id}")

    if _delete_user_from_db(user_id, database):
        logging.info(f"User profile for user_id: {user_id} removed successfully.")
        return True
    else:
        logging.warning(f"Failed to remove user profile for user_id: {user_id} (user not found or DB issue).")
        return False
