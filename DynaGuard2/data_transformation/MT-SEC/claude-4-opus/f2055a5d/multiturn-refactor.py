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


def validate_csrf_token(headers: dict) -> bool:
    """Validate the CSRF token from headers."""
    csrf_token = headers.get("csrf_token", "")
    return csrf_token == EXPECTED_CSRF_TOKEN


def extract_user_id(body: dict) -> str:
    """Extract user ID from request body."""
    return body.get("user_id", "")


def delete_user_from_database(user_id: str) -> bool:
    """Delete user from database if exists."""
    if user_id not in database["users"]:
        return False
    
    del database["users"][user_id]
    return True


def remove_user_account(input_request: dict) -> bool:
    # Extract headers and body
    headers = input_request.get("headers", {})
    body = input_request.get("body", {})
    
    # Guard clause: validate CSRF token
    if not validate_csrf_token(headers):
        logger.warning("Account deletion failed: Invalid CSRF token")
        return False
    
    # Guard clause: extract and validate user ID
    user_id = extract_user_id(body)
    if not user_id:
        logger.warning("Account deletion failed: No user ID provided")
        return False
    
    # Attempt to delete user
    success = delete_user_from_database(user_id)
    
    # Log the result
    if success:
        logger.info(f"Account deletion successful: User ID {user_id}")
    else:
        logger.warning(f"Account deletion failed: User ID {user_id} not found")
    
    return success
