"""Utilities for caching user profile information in memory.

This module exposes a simple in-memory cache (user_cache) and a helper
function to place user profile data into that cache. Caching helps avoid
repeated database lookups by keeping frequently accessed data in process
memory.

Logging:
    The module uses the standard logging library to record caching operations.
    - INFO: Successful cache writes.
    - WARNING: Input validation failures.
    - ERROR (with exception details): Unexpected failures during caching.

Global objects:
    user_cache (dict): A dictionary mapping profile IDs (str) to a dictionary
        of profile attributes. This serves as the in-memory cache.
"""

import logging
from typing import Any, Dict


# Module-level logger. A NullHandler is attached so importing this module
# does not configure logging for the application. Applications should configure
# handlers/formatters as needed.
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


# Global in-memory cache for user profiles. The cache maps a user-specific
# profile identifier (str) to a dictionary of profile attributes.
user_cache: Dict[str, Dict[str, Any]] = {}


def cache_profile_data(profile_id: str, profile_data: Dict[str, Any]) -> bool:
    """Cache user-specific profile data in memory.

    This function stores the provided profile_data under the given profile_id
    in the global user_cache dictionary. On success, it returns True; otherwise,
    it returns False.

    The function emits logs to help monitor caching behavior:
        - INFO on successful cache write (includes profile_id and number of keys)
        - WARNING if input validation fails
        - ERROR with exception details if an unexpected error occurs

    Args:
        profile_id: A unique string identifier for each user.
        profile_data: A dictionary containing various details about the user.

    Returns:
        True if the caching operation succeeds; False if it fails (for example,
        due to invalid argument types or an unexpected runtime error).

    Notes:
        - The data is stored as a shallow copy of the provided dictionary to
          minimize accidental external mutation of the cached data.
        - This cache is in-memory and per-process. It is not shared across
          multiple processes or machines and will be cleared when the process
          exits.
    """
    global user_cache

    # Validate inputs: the profile ID must be a string and the profile data
    # must be a dictionary. If validation fails, return False to signal that
    # caching did not occur.
    if not isinstance(profile_id, str) or not isinstance(profile_data, dict):
        logger.warning(
            "cache_profile_data called with invalid arguments: "
            "profile_id type=%s, profile_data type=%s",
            type(profile_id).__name__,
            type(profile_data).__name__,
        )
        return False

    try:
        key_count = len(profile_data)

        # Store a shallow copy to avoid external mutation affecting the cached
        # value. This keeps the cache stable even if the caller later modifies
        # their original dictionary.
        user_cache[profile_id] = dict(profile_data)

        logger.info(
            "Cached profile data for profile_id=%r with %d keys",
            profile_id,
            key_count,
        )
        return True
    except Exception:
        # Any unexpected failure during caching results in a False return value.
        logger.exception(
            "Failed to cache profile data for profile_id=%r", profile_id
        )
        return False
