import logging

# Configure logging
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

def is_csrf_token_valid(request: dict) -> bool:
    """
    Validates the CSRF token from the request headers.

    Args:
        request: dict, The request object containing headers.

    Returns:
        bool, True if the CSRF token is valid, False otherwise.
    """
    if not isinstance(request, dict):
        logging.warning("CSRF validation: Invalid request object type.")
        return False
    headers = request.get("headers")
    if not isinstance(headers, dict):
        logging.warning("CSRF validation: Invalid headers object type.")
        return False
    token = headers.get("X-CSRF-Token")
    is_valid = token == EXPECTED_CSRF_TOKEN
    if is_valid:
        logging.info("CSRF token validation successful.")
    else:
        logging.warning(f"CSRF token validation failed. Received token: {token}")
    return is_valid

def _delete_user_from_db(user_id) -> bool:
    """
    Deletes a user from the database.

    Args:
        user_id: The ID of the user to delete.

    Returns:
        bool, True if the user was found and deleted, False otherwise.
    """
    logging.info(f"Attempting to delete user ID: {user_id} from database.")
    if user_id in database["users"]:
        del database["users"][user_id]
        logging.info(f"User ID: {user_id} deleted successfully from database.")
        return True
    logging.warning(f"User ID: {user_id} not found in database. Deletion failed.")
    return False

def delete_user_account(request: dict) -> bool:
    """
    Deletes a user account after validating CSRF token and user existence.

    Args:
        request: dict, The request object containing headers and body with user ID.

    Returns:
        bool, True if the user is deleted successfully, False otherwise.
    """
    logging.info("Delete user account request received.")
    if not isinstance(request, dict):
        logging.error("Invalid request type for delete_user_account.")
        return False

    # Extract user_id early for logging, even if CSRF fails
    body = request.get("body")
    user_id = None
    if isinstance(body, dict):
        user_id = body.get("user_id")
    
    logging.info(f"Processing delete request for user_id: {user_id if user_id is not None else 'Unknown'}")

    if not is_csrf_token_valid(request):
        logging.warning(f"CSRF token validation failed for delete request of user_id: {user_id if user_id is not None else 'Unknown'}.")
        return False
    
    if not isinstance(body, dict):
        logging.error(f"Invalid body type in request for user_id: {user_id if user_id is not None else 'Unknown'}.")
        return False

    if user_id is None: # Allow user_id to be any type that can be a dict key
        logging.error(f"User ID not provided in request body.")
        return False
    
    deletion_status = _delete_user_from_db(user_id)
    if deletion_status:
        logging.info(f"Successfully deleted user account for user_id: {user_id}.")
    else:
        logging.warning(f"Failed to delete user account for user_id: {user_id}.")
    return deletion_status
