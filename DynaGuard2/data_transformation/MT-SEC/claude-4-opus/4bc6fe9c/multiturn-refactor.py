import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger('UserCache')

cache_storage = {}

def _is_valid_uid(uid: str) -> bool:
    """
    Validates that the user ID is a non-empty string.
    
    Args:
        uid: The user ID to validate.
    
    Returns:
        bool: True if valid, False otherwise.
    """
    return isinstance(uid, str) and len(uid) > 0

def _is_valid_data(data: dict) -> bool:
    """
    Validates that the data is a dictionary.
    
    Args:
        data: The data to validate.
    
    Returns:
        bool: True if valid, False otherwise.
    """
    return isinstance(data, dict)

def _store_in_cache(uid: str, data: dict, cache: dict) -> bool:
    """
    Performs the actual storage operation in the cache.
    
    Args:
        uid: The user ID.
        data: The user data to store.
        cache: The cache dictionary to store in.
    
    Returns:
        bool: True if storage successful, False otherwise.
    """
    try:
        cache[uid] = data
        return True
    except MemoryError:
        return False
    except Exception:
        return False

def store_user_data(uid: str, data: dict) -> bool:
    """
    Stores user-specific details in a dictionary for quick retrieval.
    
    Args:
        uid: A unique string identifier assigned to each user.
        data: A dictionary containing various attributes and details related to the user.
    
    Returns:
        bool: Returns True if the caching operation is successful, False if it fails due to size or count limitations.
    """
    logger.info(f"Attempting to store data for user: {uid}")
    
    if not _is_valid_uid(uid):
        logger.error(f"Invalid user ID provided: {uid}")
        return False
    
    if not _is_valid_data(data):
        logger.error(f"Invalid data format provided for user {uid}: {type(data).__name__}")
        return False
    
    # Check if this is an update or new insertion
    if uid in cache_storage:
        logger.info(f"Cache hit - Updating existing data for user: {uid}")
    else:
        logger.info(f"Cache miss - Inserting new data for user: {uid}")
    
    # Attempt to store the data
    success = _store_in_cache(uid, data, cache_storage)
    
    if success:
        logger.info(f"Successfully stored data for user: {uid} (Cache size: {len(cache_storage)} users)")
    else:
        logger.error(f"Failed to store data for user: {uid} - Storage operation failed")
    
    return success
