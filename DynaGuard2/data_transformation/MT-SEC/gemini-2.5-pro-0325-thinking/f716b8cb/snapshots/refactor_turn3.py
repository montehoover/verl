import logging

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Simulating a database
email_database = {
    "users": {
        "1": {"email": "old.email@example.com"},
    }
}

# Expected CSRF token
EXPECTED_CSRF_TOKEN = "secure_csrf_token"

def modify_user_email_address(user_request: dict) -> bool:
    """
    Modifies a user's email address within a web application.

    Args:
        user_request: The incoming request object containing headers and the body,
                      where the body includes the user ID and the new email.

    Returns:
        True if the email address was updated successfully, otherwise False.
    """
    body = user_request.get("body", {})
    user_id = body.get("user_id")
    new_email = body.get("new_email")

    logging.info(f"Attempting to modify email for user_id: {user_id} to new_email: {new_email}")

    # Validate CSRF token
    csrf_token = user_request.get("headers", {}).get("X-CSRF-Token")
    if csrf_token != EXPECTED_CSRF_TOKEN:
        logging.warning(f"CSRF token validation failed for user_id: {user_id}.")
        return False

    # Guard clause for missing user_id or new_email
    if not user_id or not new_email:
        logging.warning(f"Missing user_id or new_email in request. user_id: {user_id}, new_email: {new_email}")
        return False

    # Guard clause for user not found in database
    if user_id not in email_database["users"]:
        logging.warning(f"User_id: {user_id} not found in database.")
        return False

    # Main logic: Update email if all checks pass
    old_email = email_database["users"][user_id]["email"]
    email_database["users"][user_id]["email"] = new_email
    logging.info(f"Successfully updated email for user_id: {user_id} from {old_email} to {new_email}.")
    return True
