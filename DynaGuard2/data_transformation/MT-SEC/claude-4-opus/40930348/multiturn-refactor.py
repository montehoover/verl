import logging

# Configure logging
logger = logging.getLogger(__name__)

TRUSTED_USER_PROFILES = {
    "user_abc": "4e99a098d57cbf9c6f67f176106769fb",
    "user_xyz": "12e4b2f76650484030e4d447d0b5f723"
}


# Simulated function for fetching user profile data from remote service
def get_profile_from_service(user_identifier: str):
    """Simulate fetching user profile data from a remote service.
    
    Args:
        user_identifier: The identifier of the user whose profile to fetch.
        
    Returns:
        dict: A dictionary containing user profile data.
        
    Raises:
        ConnectionError: If the user profile cannot be fetched.
    """
    if user_identifier in TRUSTED_USER_PROFILES:
        return {"name": "Test User", "email": f"{user_identifier}@example.com"}
    else:
        raise ConnectionError(f"Could not fetch profile for user_id {user_identifier}")


def validate_signature(user_identifier: str, user_signature: str) -> bool:
    """Validate the cryptographic signature for a given user.
    
    Args:
        user_identifier: The user identifier.
        user_signature: The signature to validate.
        
    Returns:
        bool: True if signature is valid, False otherwise.
    """
    logger.debug(f"Validating signature for user: {user_identifier}")
    
    if user_identifier in TRUSTED_USER_PROFILES:
        expected_signature = TRUSTED_USER_PROFILES[user_identifier]
        is_valid = expected_signature == user_signature
        
        if is_valid:
            logger.debug(f"Signature validation successful for user: {user_identifier}")
        else:
            logger.warning(f"Invalid signature for user: {user_identifier}")
        
        return is_valid
    
    logger.warning(f"User not found in trusted profiles: {user_identifier}")
    return False


def fetch_user_profile(user_identifier: str):
    """Fetch user profile data from the external service.
    
    Args:
        user_identifier: The user identifier.
        
    Returns:
        dict: The user profile data.
        
    Raises:
        ConnectionError: If the profile cannot be fetched.
    """
    logger.debug(f"Attempting to fetch profile for user: {user_identifier}")
    
    try:
        profile = get_profile_from_service(user_identifier)
        logger.info(f"Successfully fetched profile for user: {user_identifier}")
        return profile
    except ConnectionError as e:
        logger.error(f"Failed to fetch profile for user: {user_identifier} - {str(e)}")
        raise


def collect_user_data(user_identifier: str, user_signature: str):
    """Retrieve a user profile from an external service and validate its cryptographic signature.
    
    This function fetches user profile data only if the provided signature
    matches the trusted signature for the given user identifier.
    
    Args:
        user_identifier: An identifier for the targeted user whose profile is being fetched.
        user_signature: The cryptographic signature attached to the profile data.
        
    Returns:
        dict or None: The user profile data only when the signature verification 
                      is successful, otherwise None.
        
    Raises:
        ConnectionError: In case of a failure to retrieve the profile.
    """
    logger.info(f"Starting user data collection for user: {user_identifier}")
    
    # Validate the signature
    if not validate_signature(user_identifier, user_signature):
        logger.warning(f"User data collection failed due to invalid signature for user: {user_identifier}")
        return None
    
    # Fetch and return the profile data
    try:
        profile = fetch_user_profile(user_identifier)
        logger.info(f"Successfully collected user data for user: {user_identifier}")
        return profile
    except ConnectionError:
        logger.error(f"User data collection failed due to connection error for user: {user_identifier}")
        raise
