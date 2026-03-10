"""
In-memory caching utilities for client/user data.

This module provides a simple dictionary-backed cache (user_cache) and a
function to add or update cached entries for rapid retrieval without
repeated database queries. It includes lightweight logging to track
caching operations.
"""

import logging
from typing import Dict, Any

# Global dictionary serving as the storage container for cached user information.
# Keys are unique client identifiers (str) and values are dictionaries with
# user-specific data.
user_cache: Dict[str, Dict[str, Any]] = {}


def cache_client_data(client_id: str, client_data: dict) -> bool:
    """
    Cache user-specific data in the in-memory storage for quick retrieval.

    This function inserts or updates the entry associated with the provided
    client_id in the global user_cache. It also logs the outcome of each
    caching operation, including the client_id and a success or failure
    message.

    Parameters:
        client_id (str): A unique string identifier for each user.
        client_data (dict): A dictionary containing various details about
            the user.

    Returns:
        bool: True if the caching operation succeeds, False otherwise.

    Notes:
        - A shallow copy of client_data is stored to avoid unintended external
          mutations affecting the cached data.
        - Logging is initialized within this function to ensure human-readable
          output without requiring global configuration.
    """
    # Initialize logging with a human-readable format within the function.
    logger = logging.getLogger(__name__)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        )
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False

    global user_cache

    try:
        # Validate input types early to ensure correct usage.
        if not isinstance(client_id, str) or not isinstance(client_data, dict):
            logger.error(
                "Failed to cache data for client_id '%s': invalid input types.",
                client_id,
            )
            return False

        # Reject empty client identifiers to avoid unusable cache entries.
        if client_id == "":
            logger.error(
                "Failed to cache data: client_id is an empty string."
            )
            return False

        # Store a shallow copy to prevent external mutations from altering the
        # cached content inadvertently.
        user_cache[client_id] = dict(client_data)
        logger.info("Cached data for client_id '%s' successfully.", client_id)

        return True

    except Exception:
        # Ensure the function always returns a bool as per its contract, even in
        # unexpected failure scenarios, and provide a descriptive log entry.
        logger.exception(
            "Failed to cache data for client_id '%s' due to an unexpected error.",
            client_id,
        )
        return False
