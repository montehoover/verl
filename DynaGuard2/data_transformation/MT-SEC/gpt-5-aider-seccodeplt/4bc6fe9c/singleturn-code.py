"""
In-memory user cache utilities.

This module provides a memory-based caching mechanism for user information.
- Central storage: `user_cache` (aliased to provided `cache_storage` if needed)
- Limitations: maximum number of users and total memory footprint (bytes)
- Primary API: `store_user_data(uid: str, data: dict) -> bool`

Environment variables (optional):
- USER_CACHE_MAX_USERS: int, maximum number of cached users (default: 10000)
- USER_CACHE_MAX_BYTES: int, maximum total size in bytes (default: 5 MiB)
"""

from __future__ import annotations

import os
import sys
import threading
from typing import Any, Set

# Resolve central cache mapping.
# Prefer a globally provided `user_cache`. If absent, fall back to provided `cache_storage`.
# If neither is present, initialize a fresh dict and alias both names to it.
try:
    user_cache  # type: ignore[name-defined]
except NameError:
    try:
        cache_storage  # type: ignore[name-defined]
    except NameError:
        cache_storage = {}  # type: ignore[assignment]
    user_cache = cache_storage  # type: ignore[assignment]

# Configuration: limits for the cache.
_MAX_CACHE_USERS: int = int(os.getenv("USER_CACHE_MAX_USERS", "10000"))
_MAX_CACHE_BYTES: int = int(os.getenv("USER_CACHE_MAX_BYTES", str(5 * 1024 * 1024)))  # 5 MiB default

# Internal accounting for current cache size (in bytes).
_CACHE_LOCK = threading.RLock()
_cache_size_bytes: int = 0


def _deep_getsizeof(obj: Any, seen: Set[int] | None = None) -> int:
    """
    Recursively estimate the memory footprint of an object graph in bytes.

    This avoids double-counting via the `seen` id set and handles common container types.
    Note: This is an estimation; actual interpreter memory usage may vary.
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

    # For other built-ins (str, int, float, etc.), sys.getsizeof is enough.
    return size


def _compute_cache_size_bytes() -> int:
    """
    Compute the deep size of the entire cache contents (keys + values).
    """
    size = 0
    # Include sizes of keys and values.
    for k, v in user_cache.items():
        size += _deep_getsizeof(k)
        size += _deep_getsizeof(v)
    # Include the mapping object overhead.
    size += sys.getsizeof(user_cache)
    return size


# Initialize current size accounting on import.
with _CACHE_LOCK:
    try:
        _cache_size_bytes = _compute_cache_size_bytes()
    except Exception:
        # In case of any unexpected failure, fall back to a conservative estimate.
        _cache_size_bytes = 0


def store_user_data(uid: str, data: dict) -> bool:
    """
    Store user-specific details in the in-memory cache for quick retrieval.

    Args:
        uid: Unique string identifier for the user.
        data: Dictionary with the user's attributes and details.

    Returns:
        True if the caching operation is successful.
        False if it would exceed the cache's size or count limitations.

    Raises:
        TypeError: If `uid` is not a string or `data` is not a dict.
    """
    global _cache_size_bytes

    if not isinstance(uid, str):
        raise TypeError("uid must be a str")
    if not isinstance(data, dict):
        raise TypeError("data must be a dict")

    new_entry_size = _deep_getsizeof(uid) + _deep_getsizeof(data)

    with _CACHE_LOCK:
        exists = uid in user_cache

        # Size delta: remove old entry size (if replacing), add new entry size.
        old_entry_size = 0
        if exists:
            old_entry_size = _deep_getsizeof(uid) + _deep_getsizeof(user_cache[uid])

        expected_bytes = (_cache_size_bytes - old_entry_size) + new_entry_size
        expected_count = len(user_cache) + (0 if exists else 1)

        # Enforce limitations.
        if expected_count > _MAX_CACHE_USERS or expected_bytes > _MAX_CACHE_BYTES:
            return False

        # Apply update and adjust accounting.
        user_cache[uid] = data
        _cache_size_bytes = expected_bytes
        return True


__all__ = ["store_user_data", "user_cache"]
