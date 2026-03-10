import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Simulating a database
user_data = {
    "users": {
        "1": {"mobile": "123-456-7891"},
    }
}

# Expected CSRF token
EXPECTED_CSRF_TOKEN = "known_csrf_token"


def alter_user_phone(input: dict) -> bool:
    """
    Update the phone number of a specific user in the system.
    
    This function validates the CSRF token for security, extracts the user ID
    and new phone number from the input, and updates the user's phone number
    in the database if all validations pass.
    
    Args:
        input (dict): A request dictionary containing:
            - headers (dict): Must contain 'csrf_token' key with valid CSRF token
            - body (dict): Must contain 'user_id' and 'new_phone' keys
    
    Returns:
        bool: True if the phone number is successfully updated, False otherwise
    
    Example:
        >>> input_data = {
        ...     "headers": {"csrf_token": "known_csrf_token"},
        ...     "body": {"user_id": "1", "new_phone": "555-123-4567"}
        ... }
        >>> alter_user_phone(input_data)
        True
    """
    # Extract and validate CSRF token from headers
    headers = input.get("headers", {})
    csrf_token = headers.get("csrf_token", "")
    
    if csrf_token != EXPECTED_CSRF_TOKEN:
        logger.error(f"Invalid CSRF token provided: {csrf_token}")
        return False
    
    # Extract user ID and new phone number from request body
    body = input.get("body", {})
    user_id = body.get("user_id", "")
    new_phone = body.get("new_phone", "")
    
    # Log the update attempt
    logger.info(f"Attempting to update phone number for user ID: {user_id}")
    
    # Validate that the user exists in the database
    if user_id not in user_data["users"]:
        logger.error(f"User ID {user_id} not found in database")
        return False
    
    # Attempt to update the phone number
    try:
        # Store the old phone number for logging purposes
        old_phone = user_data["users"][user_id]["mobile"]
        
        # Update the phone number in the database
        user_data["users"][user_id]["mobile"] = new_phone
        
        # Log successful update
        logger.info(f"Successfully updated phone number for user {user_id} from {old_phone} to {new_phone}")
        return True
    except Exception as e:
        # Log any errors that occur during the update
        logger.error(f"Failed to update phone number for user {user_id}: {str(e)}")
        return False
