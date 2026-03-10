import logging

# Configure logging
logger = logging.getLogger(__name__)

TRUSTED_USER_PROFILES = {
    "user_abc": "4e99a098d57cbf9c6f67f176106769fb",
    "user_xyz": "12e4b2f76650484030e4d447d0b5f723"
}


# Simulated function for fetching user profile data from remote service
def get_profile_from_service(uid: str):
    if uid in TRUSTED_USER_PROFILES:
        return {"name": "Test Individual", "email": f"{uid}@example.com"}
    else:
        raise ConnectionError(f"Could not fetch profile for uid {uid}")


def validate_signature(uid: str, sig: str) -> bool:
    """
    Validates the cryptographic signature for a given user ID.
    
    Args:
        uid: The unique identifier for the user
        sig: The cryptographic signature to validate
        
    Returns:
        True if the signature is valid, False otherwise
    """
    return uid in TRUSTED_USER_PROFILES and TRUSTED_USER_PROFILES[uid] == sig


def fetch_user_profile(uid: str) -> dict:
    """
    Fetches a user profile from the remote service.
    
    Args:
        uid: The unique identifier for the user
        
    Returns:
        The user profile dictionary
        
    Raises:
        ConnectionError: If the profile cannot be fetched
    """
    return get_profile_from_service(uid)


def retrieve_user_profile(uid: str, sig: str):
    """
    Retrieves a user profile from a remote service and validates its cryptographic signature.
    
    Args:
        uid: The unique identifier for the user whose profile is being fetched
        sig: The cryptographic signature associated with the user profile
        
    Returns:
        The user profile dict if the signature is valid, otherwise None
        
    Raises:
        ConnectionError: If it's not possible to fetch the profile from the remote service
    """
    logger.info(f"Attempting to retrieve profile for user: {uid}")
    
    try:
        # Fetch the profile from the remote service
        profile = fetch_user_profile(uid)
        logger.debug(f"Successfully fetched profile data for user: {uid}")
        
        # Validate the signature
        if validate_signature(uid, sig):
            logger.info(f"Profile retrieval successful for user: {uid} - signature valid")
            return profile
        else:
            logger.warning(f"Profile retrieval failed for user: {uid} - invalid signature")
            return None
            
    except ConnectionError as e:
        logger.error(f"Profile retrieval failed for user: {uid} - connection error: {str(e)}")
        raise
