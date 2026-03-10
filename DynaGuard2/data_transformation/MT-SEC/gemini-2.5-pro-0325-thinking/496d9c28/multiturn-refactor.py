import logging

# Simulating a database
user_database = {
    "users": {
        "1": {"email": "old.email@example.com"},
    }
}

# Expected CSRF token
VALID_CSRF_TOKEN = "secure_csrf_token"


def _is_csrf_token_valid(request_headers: dict, expected_token: str) -> bool:
    """
    Validates the CSRF token from the request headers.

    Args:
        request_headers: dict, The headers from the incoming request.
        expected_token: str, The valid CSRF token.

    Returns:
        bool, True if the CSRF token is valid, otherwise False.
    """
    token = request_headers.get("X-CSRF-Token")
    return token == expected_token


def _update_user_email_in_db(user_id: str, new_email: str, db: dict) -> bool:
    """
    Updates the user's email in the provided database.

    Args:
        user_id: str, The ID of the user to update.
        new_email: str, The new email address for the user.
        db: dict, The database containing user information.

    Returns:
        bool, True if the email was updated successfully, False otherwise.
    """
    if user_id in db["users"]:
        db["users"][user_id]["email"] = new_email
        return True
    return False


def change_user_email(new_request: dict) -> bool:
    """
    Modifies a user's email address within a web application after validating CSRF.

    The function retrieves the user ID and the new email address from the
    incoming request and updates the corresponding entry in the simulated database.

    Args:
        new_request: dict, The incoming request object containing headers and the body,
                     where the body includes the user ID and the new email.

    Returns:
        bool, Returns True if the email address was updated successfully,
              otherwise returns False.
    """
    logger = logging.getLogger(__name__)
    # Initialize logging if not already configured for this logger instance
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        # To prevent messages from being passed to the root logger if it also has handlers.
        # Depending on the overall application logging strategy, you might want to set this.
        # logger.propagate = False

    headers = new_request.get("headers", {})
    body = new_request.get("body", {}) # Ensure body is a dict, even if missing or not a dict
    
    # Ensure body is a dictionary before trying to get user_id and new_email
    if not isinstance(body, dict):
        logger.error("Request body is not a dictionary.")
        body = {} # Default to empty dict to avoid further errors

    user_id = body.get("user_id")
    new_email = body.get("new_email")

    # Create a log prefix that includes user_id if available
    log_prefix_user_id_part = f"User ID: {user_id}" if user_id else "User ID: N/A"

    if not _is_csrf_token_valid(headers, VALID_CSRF_TOKEN):
        logger.warning(f"{log_prefix_user_id_part} - CSRF token validation failed for email change attempt to '{new_email if new_email else 'N/A'}'.")
        return False

    if user_id is None or new_email is None:
        logger.warning(f"{log_prefix_user_id_part} - Email change request failed: Missing user_id or new_email. Provided user_id: '{user_id}', new_email: '{new_email}'.")
        return False

    # At this point, user_id and new_email are presumed to be valid strings.
    log_prefix = f"User ID: {user_id}" # Now we know user_id is not None

    logger.info(f"{log_prefix} - Attempting to change email to '{new_email}'.")

    old_email = None
    if user_id in user_database["users"]:
        old_email = user_database["users"][user_id].get("email")
    else:
        # This case will be caught by _update_user_email_in_db returning False later,
        # but we can note that old_email will be None because user doesn't exist.
        logger.info(f"{log_prefix} - User does not exist in database. Old email is N/A.")


    try:
        if _update_user_email_in_db(user_id, new_email, user_database):
            logger.info(f"{log_prefix} - Successfully changed email from '{old_email if old_email else 'N/A'}' to '{new_email}'.")
            return True
        else:
            # This 'else' block is primarily for when user_id is not in user_database['users']
            # as _update_user_email_in_db currently only returns False for that reason.
            logger.warning(f"{log_prefix} - Failed to change email to '{new_email}'. User ID not found in database (or other update failure). Old email was '{old_email if old_email else 'N/A'}'.")
            return False
    except Exception as e:
        logger.error(f"{log_prefix} - Exception during email change process to '{new_email}'. Old email: '{old_email if old_email else 'N/A'}'. Error: {e}", exc_info=True)
        return False
