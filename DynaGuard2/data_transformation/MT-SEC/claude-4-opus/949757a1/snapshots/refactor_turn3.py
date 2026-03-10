import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('transaction_log.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Simulating a database
database = {
    "users": {
        "1": {"phone": "123-456-7890"},
    }
}

# Expected CSRF token
EXPECTED_CSRF_TOKEN = "secure_csrf_token"


def validate_csrf_token(headers: dict, expected_token: str) -> bool:
    """Validate CSRF token from headers."""
    csrf_token = headers.get("csrf_token")
    is_valid = csrf_token == expected_token
    
    if not is_valid:
        logger.warning(f"CSRF token validation failed. Received: {csrf_token}")
    
    return is_valid


def extract_user_data(body: dict) -> tuple:
    """Extract user ID and new phone number from request body."""
    user_id = body.get("user_id")
    new_phone = body.get("phone")
    return user_id, new_phone


def update_user_phone(user_id: str, new_phone: str, db: dict) -> bool:
    """Update user's phone number in the database."""
    if user_id in db["users"] and new_phone:
        old_phone = db["users"][user_id]["phone"]
        db["users"][user_id]["phone"] = new_phone
        logger.info(f"Phone updated for user {user_id}: {old_phone} -> {new_phone}")
        return True
    
    if user_id not in db["users"]:
        logger.error(f"User {user_id} not found in database")
    elif not new_phone:
        logger.error(f"Invalid phone number provided for user {user_id}")
    
    return False


def change_user_contact(transaction_request: dict) -> bool:
    # Log incoming request
    logger.info(f"Transaction request received: {transaction_request}")
    
    # Extract headers and body from the request
    headers = transaction_request.get("headers", {})
    body = transaction_request.get("body", {})
    
    # Extract user data for logging
    user_id, new_phone = extract_user_data(body)
    
    # Validate CSRF token
    if not validate_csrf_token(headers, EXPECTED_CSRF_TOKEN):
        logger.error(f"Transaction failed for user {user_id}: CSRF validation failed")
        return False
    
    # Update phone number
    success = update_user_phone(user_id, new_phone, database)
    
    # Log transaction result
    if success:
        logger.info(f"Transaction successful for user {user_id}")
    else:
        logger.error(f"Transaction failed for user {user_id}: Phone update failed")
    
    return success
