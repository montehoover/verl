import logging

# Configure logging
logger = logging.getLogger(__name__)

# Global dictionary to store cached user profiles
user_cache = {}


def cache_profile_data(profile_id: str, profile_data: dict) -> bool:
    """
    Cache user profile data in memory for faster retrieval.
    
    This function stores user-specific data in a global dictionary to avoid
    frequent database queries and improve application performance.
    
    Args:
        profile_id (str): A unique string identifier for each user.
        profile_data (dict): A dictionary containing various details about the user.
    
    Returns:
        bool: True if the caching operation succeeds, False if it fails.
    """
    try:
        # Log the attempt to cache data
        logger.info(f"Attempting to cache profile data for user: {profile_id}")
        
        # Store the profile data in the global cache
        user_cache[profile_id] = profile_data
        
        # Log successful caching
        logger.info(f"Successfully cached profile data for user: {profile_id}")
        return True
        
    except Exception as e:
        # Log the error with details
        logger.error(f"Failed to cache profile data for user {profile_id}: {str(e)}")
        # Return False if any error occurs during caching
        return False
