import logging

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Simulating a database
database = {
    "users": {
        "1": {"email": "old.email@example.com"},
    }
}

# Expected CSRF token
EXPECTED_CSRF_TOKEN = "secure_csrf_token"


def _is_csrf_token_valid(token: str | None, expected_token: str) -> bool:
    """Validates the CSRF token."""
    is_valid = token == expected_token
    if is_valid:
        logging.info("CSRF token validation successful.")
    else:
        logging.warning("CSRF token validation failed.")
    return is_valid


def _update_user_email_in_db(user_id: str, new_email: str, db: dict) -> bool:
    """Updates the user's email in the database."""
    if user_id in db["users"]:
        db["users"][user_id]["email"] = new_email
        logging.info(f"Email for user_id '{user_id}' updated successfully to '{new_email}'.")
        return True
    logging.warning(f"Email update failed: user_id '{user_id}' not found.")
    return False


def change_user_email(request_data: dict) -> bool:
    """
    Updates the email address of a user.

    Args:
        request_data: A dictionary representing the HTTP request,
                      including headers and body with user data.
                      Expected structure:
                      {
                          "headers": {"X-CSRF-Token": "some_token"},
                          "body": {"user_id": "1", "new_email": "new.email@example.com"}
                      }

    Returns:
        True if the email was successfully updated, False otherwise.
    """
    logging.info("Processing change_user_email request.")
    if not isinstance(request_data, dict):
        logging.warning("Invalid request_data: not a dictionary.")
        return False

    headers = request_data.get("headers")
    body = request_data.get("body")

    if not isinstance(headers, dict) or not isinstance(body, dict):
        logging.warning("Invalid request_data: 'headers' or 'body' is not a dictionary or is missing.")
        return False

    csrf_token = headers.get("X-CSRF-Token")
    if not _is_csrf_token_valid(csrf_token, EXPECTED_CSRF_TOKEN):
        # Logging is handled within _is_csrf_token_valid
        return False

    user_id = body.get("user_id")
    new_email = body.get("new_email")

    if not user_id or not new_email:
        logging.warning(f"Missing 'user_id' or 'new_email' in request body. user_id: {user_id}, new_email: {new_email}")
        return False

    logging.info(f"Attempting to update email for user_id: '{user_id}' to '{new_email}'.")
    update_status = _update_user_email_in_db(user_id, new_email, database)
    # Logging for update_status is handled within _update_user_email_in_db
    return update_status
