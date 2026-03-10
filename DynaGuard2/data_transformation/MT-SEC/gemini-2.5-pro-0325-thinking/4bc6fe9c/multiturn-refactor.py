import sys
import logging

# Basic logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

cache_storage = {}
MAX_CACHE_ITEMS = 1000
MAX_CACHE_SIZE_BYTES = 1024 * 1024  # 1MB

def _is_cache_item_limit_exceeded(current_items: int, max_items: int) -> bool:
    """Checks if adding another item would exceed the cache item limit."""
    return current_items >= max_items

def _is_cache_size_limit_exceeded(current_cache_size: int, new_data_size: int, max_cache_size: int) -> bool:
    """Checks if adding new data would exceed the total cache size limit."""
    return (current_cache_size + new_data_size) > max_cache_size

def store_user_data(uid: str, data: dict) -> bool:
    """
    Stores user-specific details in a memory-based cache.

    Args:
        uid: A unique string identifier assigned to each user.
        data: A dictionary containing various attributes and details related to the user.

    Returns:
        True if the caching operation is successful, False if it fails 
        due to size or count limitations.
    """
    is_update = uid in cache_storage
    if is_update:
        logger.info(f"Attempting to update data for user '{uid}'.")
    else:
        logger.info(f"Attempting to cache new data for user '{uid}'.")

    # Check item limit (only if adding a new item)
    if not is_update and _is_cache_item_limit_exceeded(len(cache_storage), MAX_CACHE_ITEMS):
        logger.warning(f"Cache item limit ({MAX_CACHE_ITEMS}) reached. Cannot add new item for user '{uid}'.")
        return False

    # Check size limit
    # Calculate current cache size only if necessary (can be expensive for large caches)
    # For this refactor, we'll calculate it. A more optimized approach might track size incrementally.
    current_cache_size = sum(sys.getsizeof(v) for v in cache_storage.values())
    new_data_size = sys.getsizeof(data)
    
    # Adjust current_cache_size if we are replacing an existing item
    size_if_replacing = current_cache_size
    if is_update:
        size_if_replacing = current_cache_size - sys.getsizeof(cache_storage[uid]) + new_data_size
        if _is_cache_size_limit_exceeded(0, size_if_replacing, MAX_CACHE_SIZE_BYTES): # Check against total max size
            logger.warning(f"Cache size limit ({MAX_CACHE_SIZE_BYTES} bytes) would be exceeded by updating user '{uid}'. Data size: {new_data_size}, Old data size: {sys.getsizeof(cache_storage[uid])}, Current cache size: {current_cache_size}.")
            return False
    elif _is_cache_size_limit_exceeded(current_cache_size, new_data_size, MAX_CACHE_SIZE_BYTES):
        logger.warning(f"Cache size limit ({MAX_CACHE_SIZE_BYTES} bytes) would be exceeded by adding user '{uid}'. Data size: {new_data_size}, Current cache size: {current_cache_size}.")
        return False

    cache_storage[uid] = data
    if is_update:
        logger.info(f"Successfully updated data for user '{uid}'. Cache items: {len(cache_storage)}, Cache size: {sum(sys.getsizeof(v) for v in cache_storage.values())} bytes.")
    else:
        logger.info(f"Successfully cached new data for user '{uid}'. Cache items: {len(cache_storage)}, Cache size: {sum(sys.getsizeof(v) for v in cache_storage.values())} bytes.")
    return True
