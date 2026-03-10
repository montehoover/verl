"""
This module provides functions for caching user account information.
It includes utilities for validating account data and storing it in an in-memory cache.
"""

user_cache = {}  # Global dictionary to store user account data.


def _is_valid_account_info(account_id: str, account_data: dict) -> bool:
    """
    Validates the account ID and account data.

    Args:
        account_id: The user's account ID.
        account_data: The user's account data.

    Returns:
        True if the account information is valid (i.e., account_id is a non-empty
        string and account_data is a dictionary), False otherwise.
    """
    # Validate that account_id is a non-empty string.
    if not isinstance(account_id, str) or not account_id:
        return False
    # Validate that account_data is a dictionary.
    if not isinstance(account_data, dict):
        return False
    return True


def _store_account_info(cache: dict, account_id: str, account_data: dict) -> bool:
    """
    Stores the account data in the provided cache.

    Args:
        cache: The dictionary to store the account data in.
        account_id: The user's account ID.
        account_data: The user's account data.

    Returns:
        True if storage was successful, False if an error occurred (e.g.,
        the cache is not behaving as expected, though direct dict assignment
        is unlikely to fail unless `cache` is not a dict or due to memory issues).
    """
    try:
        # Attempt to store the account data in the cache using account_id as the key.
        cache[account_id] = account_data
        return True
    except Exception:  # Catch any potential exceptions during the dictionary assignment.
        # In a real-world application, specific exceptions should be caught and logged.
        # For example, logging.error(f"Failed to cache data for account {account_id}: {e}")
        return False


def cache_account_info(account_id: str, account_data: dict) -> bool:
    """
    Caches user-specific data in a global dictionary.

    Args:
        account_id: A unique identifier for the user (string).
        account_data: A dictionary containing details about the user.

    Returns:
        True if the user-specific data is successfully validated and cached,
        False if validation fails or if the storage operation encounters an error.
    """
    # First, validate the provided account ID and data.
    if not _is_valid_account_info(account_id, account_data):
        return False  # Return False if validation fails.

    # If validation passes, proceed to store the account information in the global user_cache.
    return _store_account_info(user_cache, account_id, account_data)
