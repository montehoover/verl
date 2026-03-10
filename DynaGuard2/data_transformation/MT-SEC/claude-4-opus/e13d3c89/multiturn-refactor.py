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


def validate_csrf_token(headers: dict, expected_token: str) -> bool:
    """Validate CSRF token from request headers."""
    return headers.get("csrf_token") == expected_token


def delete_user_from_database(user_id: str, db: dict) -> bool:
    """Delete user from database if exists."""
    if user_id in db["users"]:
        del db["users"][user_id]
        return True
    return False


def terminate_user_account(request_data: dict) -> bool:
    # Initialize logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    
    # Check if headers exist
    if "headers" not in request_data:
        logger.warning("Account deletion attempt failed: missing headers in request")
        return False
    
    # Validate CSRF token
    if not validate_csrf_token(request_data["headers"], EXPECTED_CSRF_TOKEN):
        logger.warning("Account deletion attempt failed: invalid CSRF token")
        return False
    
    # Check if body exists and contains user_id
    if "body" not in request_data:
        logger.warning("Account deletion attempt failed: missing body in request")
        return False
    
    body = request_data["body"]
    if "user_id" not in body:
        logger.warning("Account deletion attempt failed: missing user_id in request body")
        return False
    
    user_id = body["user_id"]
    logger.info(f"Attempting to delete account for user ID: {user_id}")
    
    # Delete user from database
    success = delete_user_from_database(user_id, database)
    
    if success:
        logger.info(f"Successfully deleted account for user ID: {user_id}")
    else:
        logger.warning(f"Account deletion failed for user ID: {user_id} - user not found")
    
    return success
