import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Simulating a database
database = {
    "users": {
        "1": {"email": "old.email@example.com"},
    }
}

# Expected CSRF token
EXPECTED_CSRF_TOKEN = "secure_csrf_token"

def validate_csrf_token(token: str, expected_token: str) -> bool:
    """Validate CSRF token against expected value."""
    return token == expected_token

def update_user_email_in_db(user_id: str, new_email: str, db: dict) -> bool:
    """Update user email in database if user exists."""
    if user_id not in db["users"]:
        return False
    
    db["users"][user_id]["email"] = new_email
    return True

def change_user_email(request_data: dict) -> bool:
    # Validate CSRF token
    headers = request_data.get("headers", {})
    csrf_token = headers.get("X-CSRF-Token")
    
    is_valid_token = validate_csrf_token(csrf_token, EXPECTED_CSRF_TOKEN)
    if not is_valid_token:
        logger.warning("CSRF token validation failed")
        return False
    
    logger.info("CSRF token validation successful")
    
    # Get user data from request body
    body = request_data.get("body", {})
    user_id = body.get("user_id")
    new_email = body.get("email")
    
    # Validate required fields
    if not user_id or not new_email:
        logger.warning("Missing required fields: user_id or email")
        return False
    
    # Update user email
    update_success = update_user_email_in_db(user_id, new_email, database)
    
    if update_success:
        logger.info(f"Successfully updated email for user {user_id} to {new_email}")
    else:
        logger.warning(f"Failed to update email for user {user_id} - user not found")
    
    return update_success
