import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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


def validate_csrf_token(token: str, expected_token: str) -> bool:
    """Validate CSRF token against expected value."""
    return token == expected_token


def delete_user_from_database(user_id: str, db: dict) -> bool:
    """Delete user from database if exists."""
    if user_id and user_id in db["users"]:
        del db["users"][user_id]
        return True
    return False


def delete_user_account(request: dict) -> bool:
    # Extract CSRF token from headers
    csrf_token = request.get("headers", {}).get("X-CSRF-Token")
    
    # Extract user ID from request body
    user_id = request.get("body", {}).get("user_id")
    
    logger.info(f"Deletion attempt for user ID: {user_id}")
    
    # Verify CSRF token
    csrf_valid = validate_csrf_token(csrf_token, EXPECTED_CSRF_TOKEN)
    if not csrf_valid:
        logger.warning(f"Invalid CSRF token for deletion attempt of user ID: {user_id}")
        return False
    
    logger.info(f"Valid CSRF token for user ID: {user_id}")
    
    # Delete the user
    deletion_success = delete_user_from_database(user_id, database)
    
    if deletion_success:
        logger.info(f"Successfully deleted user ID: {user_id}")
    else:
        logger.warning(f"Failed to delete user ID: {user_id} - User not found")
    
    return deletion_success
