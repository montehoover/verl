import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Simulating a database
db_users = {
    "users": {
        "1": {"telephone": "123-556-7890"},
    }
}

# Expected CSRF token
EXPECTED_CSRF_TOKEN = "csrf_token_secured"


def change_user_phone(payload: dict) -> bool:
    # Check if headers exist and contain CSRF token
    if "headers" not in payload:
        logger.warning("Phone update failed: Missing headers in payload")
        return False
    
    headers = payload.get("headers", {})
    if headers.get("csrf_token") != EXPECTED_CSRF_TOKEN:
        logger.warning("Phone update failed: Invalid CSRF token")
        return False
    
    # Check if body exists and contains required fields
    if "body" not in payload:
        logger.warning("Phone update failed: Missing body in payload")
        return False
    
    body = payload.get("body", {})
    user_id = body.get("user_id")
    new_phone = body.get("new_phone")
    
    # Validate user_id and new_phone exist
    if user_id is None or new_phone is None:
        logger.warning(f"Phone update failed: Missing user_id or new_phone in request body. user_id={user_id}, new_phone provided={new_phone is not None}")
        return False
    
    # Check if user exists in database
    if user_id not in db_users["users"]:
        logger.warning(f"Phone update failed: User ID {user_id} not found in database")
        return False
    
    # Update the phone number
    try:
        old_phone = db_users["users"][user_id]["telephone"]
        db_users["users"][user_id]["telephone"] = new_phone
        logger.info(f"Phone update successful: User ID {user_id} phone changed from {old_phone} to {new_phone}")
        return True
    except Exception as e:
        logger.error(f"Phone update failed: Error updating phone for User ID {user_id}. Exception: {str(e)}")
        return False
