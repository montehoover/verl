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

def _is_csrf_token_valid(request: dict) -> bool:
    """
    Validates the CSRF token from the request headers.

    Args:
        request: dict, An object representing the HTTP request.

    Returns:
        bool, True if the CSRF token is valid, False otherwise.
    """
    if not isinstance(request, dict) or 'headers' not in request:
        return False
    token = request['headers'].get('X-CSRF-Token')
    return token == EXPECTED_CSRF_TOKEN

def _delete_user(user_id: str) -> bool:
    """
    Deletes a user from the database.

    Args:
        user_id: str, The ID of the user to delete.

    Returns:
        bool, True if the user was deleted successfully, False otherwise.
    """
    if user_id in database["users"]:
        del database["users"][user_id]
        return True
    return False

def remove_account(request: dict) -> bool:
    """
    Removes a user account from the system.

    Args:
        request: dict, An object representing the HTTP request,
                       containing headers and body with the user ID.

    Returns:
        bool, True if the user account is deleted successfully, False otherwise.
    """
    # Initialize logging
    logger = logging.getLogger(__name__)
    if not logger.handlers: # Avoid adding multiple handlers if called multiple times
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)

    if not isinstance(request, dict) or \
       'body' not in request or \
       'user_id' not in request['body']:
        logger.error("Invalid request format for user_id extraction.")
        return False

    user_id = request['body']['user_id']
    logger.info(f"Attempting to remove account for user_id: {user_id}")

    if not _is_csrf_token_valid(request):
        logger.warning(f"CSRF token validation failed for user_id: {user_id}")
        return False  # CSRF token invalid or missing
    logger.info(f"CSRF token validated successfully for user_id: {user_id}")

    deleted = _delete_user(user_id)
    if deleted:
        logger.info(f"Successfully removed account for user_id: {user_id}")
    else:
        logger.error(f"Failed to remove account for user_id: {user_id}. User not found.")
    return deleted
