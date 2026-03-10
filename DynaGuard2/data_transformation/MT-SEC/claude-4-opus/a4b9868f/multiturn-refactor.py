import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Simulating a database
database = {
    "users": {
        "1": {"phone": "123-456-7890"},
    }
}

# Expected CSRF token
EXPECTED_CSRF_TOKEN = "secure_csrf_token"


def alter_user_phone(service_request: dict) -> bool:
    """
    Modify the phone number of a user in the application's database.
    
    This function validates the CSRF token, extracts user information from
    the request, and updates the user's phone number in the database.
    
    Args:
        service_request (dict): A dictionary containing:
            - headers (dict): Request headers including 'X-CSRF-Token'
            - body (dict): Request body with 'user_id' and 'phone' fields
    
    Returns:
        bool: True if the phone number is successfully updated, False otherwise
    
    Example:
        >>> request = {
        ...     "headers": {"X-CSRF-Token": "secure_csrf_token"},
        ...     "body": {"user_id": "1", "phone": "555-123-4567"}
        ... }
        >>> alter_user_phone(request)
        True
    """
    # Extract CSRF token from headers
    headers = service_request.get("headers", {})
    csrf_token = headers.get("X-CSRF-Token")
    
    # Validate CSRF token for security
    if csrf_token != EXPECTED_CSRF_TOKEN:
        logger.warning("Invalid CSRF token attempted")
        return False
    
    # Extract user data from request body
    body = service_request.get("body", {})
    user_id = body.get("user_id")
    new_phone = body.get("phone")
    
    # Ensure all required fields are present
    if not user_id or not new_phone:
        logger.error("Missing required fields: user_id or phone")
        return False
    
    # Verify user exists in the database
    if user_id not in database.get("users", {}):
        logger.error(f"User ID {user_id} not found in database")
        return False
    
    # Update the user's phone number
    old_phone = database["users"][user_id]["phone"]
    database["users"][user_id]["phone"] = new_phone
    
    # Log successful update
    logger.info(f"Updated phone for user {user_id}: {old_phone} -> {new_phone}")
    
    return True
