from typing import Any, Dict
import sys
import threading

# Global cache storage (provided setup)
cache_storage: Dict[str, Dict[str, Any]] = {}

# Cache capacity constraints
MAX_CACHE_ITEMS: int = 10000  # Maximum number of users to cache
MAX_CACHE_SIZE_BYTES: int = 25 * 1024 * 1024  # 25 MiB maximum total cache size

# Internal tracking for efficient size checks
_cache_size_bytes: int = 0
_cache_lock = threading.Lock()


def _deep_getsizeof(obj: Any, seen: set[int] | None = None) -> int:
    """
    Recursively estimate the memory footprint of a Python object in bytes.
    Avoids double-counting by tracking seen object ids.
    """
    if seen is None:
        seen = set()
    obj_id = id(obj)
    if obj_id in seen:
        return 0
    seen.add(obj_id)

    size = sys.getsizeof(obj)

    if isinstance(obj, dict):
        for k, v in obj.items():
            size += _deep_getsizeof(k, seen)
            size += _deep_getsizeof(v, seen)
    elif isinstance(obj, (list, tuple, set, frozenset)):
        for item in obj:
            size += _deep_getsizeof(item, seen)

    return size


def store_user_data(uid: str, data: dict) -> bool:
    """
    Stores user-specific details in an in-memory cache for quick retrieval.

    Args:
        uid (str): A unique string identifier assigned to each user.
        data (dict): A dictionary containing various attributes and details related to the user.

    Returns:
        bool: True if the caching operation is successful,
              False if it fails due to size or count limitations.
    """
    global _cache_size_bytes

    # Basic normalization to reduce unexpected failures on non-dict mappings
    if not isinstance(data, dict):
        try:
            data = dict(data)  # Attempt to coerce mapping-like objects
        except Exception:
            return False

    # Estimate size of new/updated entry (key + value)
    new_entry_size = _deep_getsizeof(uid) + _deep_getsizeof(data)

    with _cache_lock:
        is_update = uid in cache_storage
        old_entry_size = 0
        if is_update:
            # Size of existing entry to compute delta
            old_entry_size = _deep_getsizeof(uid) + _deep_getsizeof(cache_storage[uid])

        # Enforce count limit (only when inserting a new entry)
        projected_count = len(cache_storage) + (0 if is_update else 1)
        if projected_count > MAX_CACHE_ITEMS:
            return False

        # Enforce size limit using a running total and the entry delta
        projected_size = _cache_size_bytes + (new_entry_size - old_entry_size)
        if projected_size > MAX_CACHE_SIZE_BYTES:
            return False

        # Commit to cache and update size tracker
        cache_storage[uid] = data
        _cache_size_bytes = projected_size
        return True
