import logging

# Configure logging
logger = logging.getLogger(__name__)

TRUSTED_USER_PROFILES = {
    "user_abc": "4e99a098d57cbf9c6f67f176106769fb",
    "user_xyz": "12e4b2f76650484030e4d447d0b5f723"
}


# Simulated function for fetching user profile data from remote service
def get_profile_from_service(uuid: str):
    if uuid in TRUSTED_USER_PROFILES:
        return {"name": "Test User", "email": f"{uuid}@example.com"}
    else:
        raise ConnectionError(f"Could not fetch profile for user_id {uuid}")


def verify_signature(uuid: str, sig_hash: str) -> bool:
    """Verify if the provided signature matches the trusted signature for the user."""
    return uuid in TRUSTED_USER_PROFILES and TRUSTED_USER_PROFILES[uuid] == sig_hash


def fetch_user_profile(uuid: str) -> dict:
    """Fetch user profile data from the remote service."""
    return get_profile_from_service(uuid)


def collect_user_data(uuid: str, sig_hash: str):
    logger.info(f"Starting user data collection for user_id: {uuid}")
    
    try:
        logger.debug(f"Fetching profile data for user_id: {uuid}")
        profile_data = fetch_user_profile(uuid)
        logger.debug(f"Successfully fetched profile data for user_id: {uuid}")
        
        logger.debug(f"Verifying signature for user_id: {uuid}")
        if verify_signature(uuid, sig_hash):
            logger.info(f"Signature verification successful for user_id: {uuid}")
            return profile_data
        else:
            logger.warning(f"Signature verification failed for user_id: {uuid}")
            return None
            
    except ConnectionError as e:
        logger.error(f"Connection error while fetching profile for user_id: {uuid} - {str(e)}")
        raise
