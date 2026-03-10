"""
Module providing a simple user cache and functions to store account info.

The cache is a global dictionary 'user_cache' used to store user-specific
data for rapid retrieval and to reduce repeated database queries.
"""

# Global cache storage used across the module.
user_cache = {}


def is_valid_account_info(account_id: str, account_data: dict) -> bool:
    """
    Validate inputs for caching account information.

    Args:
        account_id (str): Unique identifier for the user.
        account_data (dict): Dictionary containing user details to cache.

    Returns:
        bool: True if inputs are valid, False otherwise.

    Notes:
        - 'account_id' must be a non-empty string.
        - 'account_data' must be a dictionary.
    """
    # Basic type and value checks help ensure cache integrity and predictable
    # behavior in downstream lookups.
    if not isinstance(account_id, str) or not account_id:
        return False

    if not isinstance(account_data, dict):
        return False

    return True


def store_account_info(cache: dict, account_id: str, account_data: dict) -> dict:
    """
    Produce a new cache dictionary with the account data stored.

    This function is pure and does not mutate the provided cache; it creates
    and returns a new dictionary with the updated entry.

    Args:
        cache (dict): Existing cache mapping.
        account_id (str): Unique identifier for the user.
        account_data (dict): Dictionary containing user details.

    Returns:
        dict: A new cache with the updated/inserted account entry.
    """
    # Create a shallow copy to avoid mutating the original cache.
    new_cache = dict(cache)

    # Insert or update the user's account data keyed by 'account_id'.
    new_cache[account_id] = account_data

    return new_cache


def cache_account_info(account_id: str, account_data: dict) -> bool:
    """
    Cache user-specific data keyed by a unique account identifier.

    Args:
        account_id (str): Unique identifier for the user.
        account_data (dict): Dictionary containing user details to cache.

    Returns:
        bool: True if caching succeeds, False otherwise.
    """
    # Validate the inputs before attempting to store them.
    if not is_valid_account_info(account_id, account_data):
        return False

    try:
        # Update the global cache by replacing it with a new, updated mapping.
        # Using a pure helper makes unit testing simple and avoids side effects.
        global user_cache
        user_cache = store_account_info(user_cache, account_id, account_data)
        return True
    except Exception:
        # Any unexpected failure should return False per contract.
        # Avoid exposing internal exceptions at this layer.
        return False
