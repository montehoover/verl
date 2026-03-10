import logging
from datetime import datetime

# Simulating a database
database = {
    "users": {
        "1": {"phone": "123-456-7890"},
    }
}

# Expected CSRF token
EXPECTED_CSRF_TOKEN = "secure_csrf_token"

# Configure logging
logging.basicConfig(
    filename='phone_changes.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)


def validate_csrf_token(csrf_token: str) -> bool:
    """Validate the CSRF token against the expected value."""
    return csrf_token == EXPECTED_CSRF_TOKEN


def update_phone_in_database(user_id: str, new_phone: str) -> bool:
    """Update the phone number for a user in the database."""
    try:
        if user_id in database.get("users", {}):
            old_phone = database["users"][user_id]["phone"]
            database["users"][user_id]["phone"] = new_phone
            
            # Log successful update
            logging.info(f"Phone update successful - User ID: {user_id}, Old phone: {old_phone}, New phone: {new_phone}")
            return True
        
        # Log failed update - user not found
        logging.warning(f"Phone update failed - User ID: {user_id} not found in database")
        return False
    except Exception as e:
        # Log exception
        logging.error(f"Phone update failed - User ID: {user_id}, Error: {str(e)}")
        return False


def change_user_phone(request_details: dict) -> bool:
    # Extract CSRF token from headers
    headers = request_details.get("headers", {})
    csrf_token = headers.get("X-CSRF-Token")
    
    # Validate CSRF token
    if not validate_csrf_token(csrf_token):
        logging.warning("Phone update failed - Invalid CSRF token")
        return False
    
    # Extract user ID and new phone number from body
    body = request_details.get("body", {})
    user_id = body.get("user_id")
    new_phone = body.get("new_phone")
    
    # Check if required fields are present
    if not user_id or not new_phone:
        logging.warning(f"Phone update failed - Missing required fields. User ID: {user_id}, New phone: {new_phone}")
        return False
    
    # Log the request
    logging.info(f"Phone update requested - User ID: {user_id}, New phone: {new_phone}")
    
    # Update the phone number
    return update_phone_in_database(user_id, new_phone)
