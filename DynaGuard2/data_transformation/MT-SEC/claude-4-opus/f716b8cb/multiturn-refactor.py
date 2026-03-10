import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Simulating a database
email_database = {
    "users": {
        "1": {"email": "old.email@example.com"},
    }
}

# Expected CSRF token
EXPECTED_CSRF_TOKEN = "secure_csrf_token"

def modify_user_email_address(user_request: dict) -> bool:
    # Log incoming request
    logger.info(f"Received email update request")
    
    # Extract CSRF token from headers
    csrf_token = user_request.get("headers", {}).get("X-CSRF-Token")
    
    # Validate CSRF token
    if csrf_token != EXPECTED_CSRF_TOKEN:
        logger.warning("Invalid CSRF token provided")
        return False
    
    # Extract user_id and new_email from request body
    body = user_request.get("body", {})
    user_id = body.get("user_id")
    new_email = body.get("new_email")
    
    # Log extracted information
    logger.info(f"Request for user_id: {user_id}, new_email: {new_email}")
    
    # Check if user_id and new_email are provided
    if not user_id or not new_email:
        logger.error("Missing required fields: user_id or new_email")
        return False
    
    # Check if user exists in database
    if user_id not in email_database["users"]:
        logger.error(f"User {user_id} not found in database")
        return False
    
    # Get old email for logging
    old_email = email_database["users"][user_id]["email"]
    
    # Update email address
    email_database["users"][user_id]["email"] = new_email
    logger.info(f"Successfully updated email for user {user_id} from {old_email} to {new_email}")
    
    return True
