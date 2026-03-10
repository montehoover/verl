import sys

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
    if uid in cache_storage: # Optional: Overwrite existing data or return False/handle differently
        pass # Current behavior is to overwrite

    # Check item limit
    if not (uid in cache_storage) and _is_cache_item_limit_exceeded(len(cache_storage), MAX_CACHE_ITEMS):
        # Potentially log this event: print(f"Cache item limit ({MAX_CACHE_ITEMS}) reached. Cannot add new item.")
        return False

    # Check size limit
    # Calculate current cache size only if necessary (can be expensive for large caches)
    # For this refactor, we'll calculate it. A more optimized approach might track size incrementally.
    current_cache_size = sum(sys.getsizeof(v) for v in cache_storage.values())
    new_data_size = sys.getsizeof(data)
    
    # Adjust current_cache_size if we are replacing an existing item
    size_if_replacing = current_cache_size
    if uid in cache_storage:
        size_if_replacing = current_cache_size - sys.getsizeof(cache_storage[uid]) + new_data_size
        if _is_cache_size_limit_exceeded(0, size_if_replacing, MAX_CACHE_SIZE_BYTES): # Check against total max size
            # Potentially log this event: print(f"Cache size limit ({MAX_CACHE_SIZE_BYTES} bytes) reached. Cannot update item '{uid}'.")
            return False
    elif _is_cache_size_limit_exceeded(current_cache_size, new_data_size, MAX_CACHE_SIZE_BYTES):
        # Potentially log this event: print(f"Cache size limit ({MAX_CACHE_SIZE_BYTES} bytes) reached. Cannot add new item '{uid}'.")
        return False

    cache_storage[uid] = data
    return True
