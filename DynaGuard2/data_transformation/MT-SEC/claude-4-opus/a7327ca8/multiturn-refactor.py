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


def validate_csrf_token(token: str) -> bool:
    """
    Validate the CSRF token against the expected value.
    
    Args:
        token: str, The CSRF token to validate.
        
    Returns:
        bool, True if the token is valid, False otherwise.
    """
    return token == EXPECTED_CSRF_TOKEN


def delete_user_from_database(user_id: str) -> bool:
    """
    Delete a user from the database.
    
    Args:
        user_id: str, The ID of the user to delete.
        
    Returns:
        bool, True if the user was deleted, False if user not found.
    """
    if user_id in database["users"]:
        del database["users"][user_id]
        return True
    return False


def remove_account(request: dict) -> bool:
    """
    Remove a user account from the system.
    
    Args:
        request: dict, An object representing the HTTP request, containing headers and body with the user ID.
        
    Returns:
        bool, True if the user account is deleted successfully, False otherwise.
    """
    # Initialize logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    logger = logging.getLogger(__name__)
    
    try:
        # Extract CSRF token from headers
        headers = request.get("headers", {})
        csrf_token = headers.get("X-CSRF-Token", "")
        
        # Verify CSRF token
        if not validate_csrf_token(csrf_token):
            logger.warning("Account removal failed: Invalid CSRF token")
            return False
            
        # Extract user ID from request body
        body = request.get("body", {})
        user_id = body.get("user_id", "")
        
        if not user_id:
            logger.warning("Account removal failed: No user ID provided")
            return False
        
        logger.info(f"Attempting to remove account for user ID: {user_id}")
        
        # Delete user from database
        result = delete_user_from_database(user_id)
        
        if result:
            logger.info(f"Successfully removed account for user ID: {user_id}")
        else:
            logger.warning(f"Account removal failed: User ID {user_id} not found")
            
        return result
        
    except Exception as e:
        logger.error(f"Account removal failed due to unexpected error: {str(e)}")
        return False
