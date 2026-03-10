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


# -------------------- Pure helper functions -------------------- #

def normalize_user_data(data: Any) -> tuple[bool, dict]:
    """
    Attempts to coerce the provided data into a plain dict.

    Returns:
        tuple[bool, dict]: (success flag, coerced dict if success else empty dict)
    """
    if isinstance(data, dict):
        return True, data
    try:
        return True, dict(data)
    except Exception:
        return False, {}


def entry_size(uid: str, data: dict) -> int:
    """
    Calculates the memory footprint for a cache entry consisting of the UID and its data dict.
    """
    return _deep_getsizeof(uid) + _deep_getsizeof(data)


def compute_new_old_sizes(uid: str, new_data: dict, cache: Dict[str, Dict[str, Any]]) -> tuple[int, int, bool]:
    """
    Computes the sizes for the new and (if present) existing cache entries.
    Returns:
        (new_entry_size, old_entry_size, is_update)
    """
    is_update = uid in cache
    new_size = entry_size(uid, new_data)
    old_size = entry_size(uid, cache[uid]) if is_update else 0
    return new_size, old_size, is_update


def compute_projected_count(cache_len: int, is_update: bool) -> int:
    """
    Computes the projected number of cache items after an insert/update.
    """
    return cache_len + (0 if is_update else 1)


def compute_projected_size(current_size: int, new_size: int, old_size: int) -> int:
    """
    Computes the projected total cache size after applying the size delta.
    """
    return current_size + (new_size - old_size)


def within_limits(projected_count: int, projected_size: int, max_items: int, max_size: int) -> bool:
    """
    Validates projected cache state against item count and total size limits.
    """
    if projected_count > max_items:
        return False
    if projected_size > max_size:
        return False
    return True


# -------------------- Public API -------------------- #

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

    ok, normalized = normalize_user_data(data)
    if not ok:
        return False

    with _cache_lock:
        new_size, old_size, is_update = compute_new_old_sizes(uid, normalized, cache_storage)
        projected_count = compute_projected_count(len(cache_storage), is_update)
        projected_size = compute_projected_size(_cache_size_bytes, new_size, old_size)

        if not within_limits(projected_count, projected_size, MAX_CACHE_ITEMS, MAX_CACHE_SIZE_BYTES):
            return False

        cache_storage[uid] = normalized
        _cache_size_bytes = projected_size
        return True
