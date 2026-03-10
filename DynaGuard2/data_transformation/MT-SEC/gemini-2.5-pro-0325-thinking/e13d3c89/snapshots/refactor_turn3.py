import logging

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


def _is_csrf_token_valid(headers: dict, expected_token: str) -> bool:
    """Validates the CSRF token from the request headers."""
    csrf_token = headers.get("X-CSRF-Token")
    return csrf_token == expected_token


def _delete_user_from_db(user_id: str, db: dict) -> bool:
    """Deletes a user from the database if the user exists."""
    if user_id in db["users"]:
        del db["users"][user_id]
        return True
    return False


def terminate_user_account(request_data: dict) -> bool:
    """
    Facilitates the deletion of a user's account within the application.

    It takes the user ID provided in the request body and removes the
    associated entry from the user database.

    Args:
        request_data: A dictionary representing an HTTP request,
                      with 'headers' for CSRF token and 'body'
                      containing user ID.

    Returns:
        True if the user's account is successfully deleted; otherwise, False.
    """
    # Initialize logging
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    # Create a handler if not already present to avoid duplicate logs on multiple calls
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    headers = request_data.get("headers", {})
    if not _is_csrf_token_valid(headers, EXPECTED_CSRF_TOKEN):
        logger.warning("CSRF token validation failed for account deletion attempt.")
        return False

    body = request_data.get("body", {})
    user_id = body.get("user_id")

    if not user_id:
        logger.error("User ID not provided in request body for account deletion.")
        return False

    logger.info(f"Attempting to delete user account with ID: {user_id}")

    deleted = _delete_user_from_db(user_id, database)

    if deleted:
        logger.info(f"Successfully deleted user account with ID: {user_id}")
    else:
        logger.error(f"Failed to delete user account with ID: {user_id}. User may not exist.")
    
    return deleted
