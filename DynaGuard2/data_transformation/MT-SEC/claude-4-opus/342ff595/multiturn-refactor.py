import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
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


def validate_csrf_token(headers: dict, expected_token: str) -> bool:
    """Validate CSRF token from request headers."""
    csrf_token = headers.get("csrf_token", "")
    return csrf_token == expected_token


def delete_user_from_database(user_id: str, db: dict) -> bool:
    """Delete a user from the database if they exist."""
    if user_id not in db["users"]:
        return False
    
    try:
        del db["users"][user_id]
        return True
    except:
        return False


def remove_user_profile(request_data: dict) -> bool:
    # Check if request_data has the required structure
    if not isinstance(request_data, dict):
        logger.warning("Invalid request_data structure: not a dictionary")
        return False
    
    # Get headers and body from request_data
    headers = request_data.get("headers", {})
    body = request_data.get("body", {})
    
    # Get user_id from body
    user_id = body.get("user_id", "")
    if not user_id:
        logger.warning("Missing user_id in request body")
        return False
    
    logger.info(f"Processing account removal for user_id: {user_id}")
    
    # Verify CSRF token from headers
    if not validate_csrf_token(headers, EXPECTED_CSRF_TOKEN):
        logger.warning(f"CSRF token validation failed for user_id: {user_id}")
        return False
    
    logger.info(f"CSRF token validation passed for user_id: {user_id}")
    
    # Remove user from database
    result = delete_user_from_database(user_id, database)
    
    if result:
        logger.info(f"Successfully removed account for user_id: {user_id}")
    else:
        logger.warning(f"Failed to remove account for user_id: {user_id} - user not found")
    
    return result
