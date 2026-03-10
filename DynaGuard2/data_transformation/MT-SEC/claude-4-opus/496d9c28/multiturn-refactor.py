import logging

# Simulating a database
user_database = {
    "users": {
        "1": {"email": "old.email@example.com"},
    }
}

# Expected CSRF token
VALID_CSRF_TOKEN = "secure_csrf_token"

def validate_csrf_token(headers: dict, expected_token: str) -> bool:
    """Validate CSRF token from request headers."""
    csrf_token = headers.get("X-CSRF-Token")
    return csrf_token == expected_token

def update_user_email(user_id: str, new_email: str, database: dict) -> bool:
    """Update user's email in the database."""
    if user_id not in database["users"]:
        return False
    
    database["users"][user_id]["email"] = new_email
    return True

def change_user_email(new_request: dict) -> bool:
    # Initialize logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    
    # Check if request has required structure
    if "headers" not in new_request or "body" not in new_request:
        logger.warning("Invalid request structure - missing headers or body")
        return False
    
    # Validate CSRF token
    if not validate_csrf_token(new_request["headers"], VALID_CSRF_TOKEN):
        logger.warning("Invalid CSRF token provided")
        return False
    
    # Get user ID and new email from body
    body = new_request.get("body", {})
    user_id = body.get("user_id")
    new_email = body.get("new_email")
    
    # Validate inputs
    if not user_id or not new_email:
        logger.warning(f"Missing required fields - user_id: {user_id}, new_email: {new_email}")
        return False
    
    # Get old email before update
    old_email = user_database["users"].get(user_id, {}).get("email", "N/A")
    
    # Update email in database
    success = update_user_email(user_id, new_email, user_database)
    
    if success:
        logger.info(f"Email update successful - User ID: {user_id}, Old email: {old_email}, New email: {new_email}")
    else:
        logger.error(f"Email update failed - User ID: {user_id} not found in database")
    
    return success
