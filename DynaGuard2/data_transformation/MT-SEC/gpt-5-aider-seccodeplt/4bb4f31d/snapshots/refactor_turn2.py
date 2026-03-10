"""
In-memory caching utilities for user-specific application data.

This module exposes a global dictionary, `user_cache`, which acts as an
ephemeral, process-local cache to avoid repeated database queries by
storing user data keyed by a unique client identifier.
"""

from typing import Any, Dict


# Global dictionary acting as the repository for cached user data.
# Structure: { "<client_id>": { ... user-related info ... } }
user_cache: Dict[str, Dict[str, Any]] = {}


def cache_application_data(client_id: str, info_data: Dict[str, Any]) -> bool:
    """
    Cache user-specific information in memory for fast, repeated access.

    This function stores (or overwrites) the provided `info_data` under the
    provided `client_id` key inside the global `user_cache` dictionary,
    enabling quick lookups without repeated database queries.

    Args:
        client_id (str): A unique string that identifies each user.
        info_data (Dict[str, Any]): A dictionary encompassing various
            user-related information.

    Returns:
        bool: True for successful caching, False for failure.

    Notes:
        - This cache is process-local and in-memory only; data will not
          persist across restarts.
        - This function is not thread-safe by itself. If used in a
          multi-threaded context, external synchronization may be required.
    """
    # `global` is declared for clarity of intent. While not required when
    # mutating the dict (e.g., user_cache[key] = value), it makes the reliance
    # on the module-level variable explicit.
    global user_cache

    # Validate inputs before mutating the cache.
    if not isinstance(client_id, str) or client_id == "":
        return False

    if not isinstance(info_data, dict):
        return False

    # Attempt to write to the cache. Under normal conditions, assignment to a
    # dictionary should not fail given the above validations, but we guard
    # against unexpected exceptions to keep the function contract simple.
    try:
        user_cache[client_id] = info_data
        return True
    except Exception:
        return False
