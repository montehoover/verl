"""
In-memory caching utilities for client/user data.

This module provides a simple dictionary-backed cache (user_cache) and a
function to add or update cached entries for rapid retrieval without
repeated database queries.
"""

from typing import Dict, Any

# Global dictionary serving as the storage container for cached user information.
# Keys are unique client identifiers (str) and values are dictionaries with
# user-specific data.
user_cache: Dict[str, Dict[str, Any]] = {}


def cache_client_data(client_id: str, client_data: dict) -> bool:
    """
    Cache user-specific data in the in-memory storage for quick retrieval.

    This function inserts or updates the entry associated with the provided
    client_id in the global user_cache.

    Parameters:
        client_id (str): A unique string identifier for each user.
        client_data (dict): A dictionary containing various details about
            the user.

    Returns:
        bool: True if the caching operation succeeds, False otherwise.

    Notes:
        - A shallow copy of client_data is stored to avoid unintended external
          mutations affecting the cached data.
        - The function catches all exceptions to preserve the boolean return
          contract and avoid propagating unexpected errors.
    """
    global user_cache

    try:
        # Validate input types early to ensure correct usage.
        if not isinstance(client_id, str) or not isinstance(client_data, dict):
            return False

        # Reject empty client identifiers to avoid unusable cache entries.
        if client_id == "":
            return False

        # Store a shallow copy to prevent external mutations from altering the
        # cached content inadvertently.
        user_cache[client_id] = dict(client_data)

        return True

    except Exception:
        # Intentionally broad catch: ensure the function always returns a bool
        # as per its contract, even in unexpected failure scenarios.
        return False
