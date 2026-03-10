"""
Module providing a simple user cache and a function to store account info.

The cache is a global dictionary `user_cache` used to store user-specific data
for rapid retrieval, reducing repeated database queries.
"""

# Global cache storage
user_cache = {}

def is_valid_account_info(account_id: str, account_data: dict) -> bool:
    """
    Validate the inputs for caching account information.

    Args:
        account_id (str): Unique identifier for the user.
        account_data (dict): Dictionary containing user details to cache.

    Returns:
        bool: True if inputs are valid, False otherwise.
    """
    if not isinstance(account_id, str) or not account_id:
        return False
    if not isinstance(account_data, dict):
        return False
    return True

def store_account_info(cache: dict, account_id: str, account_data: dict) -> dict:
    """
    Produce a new cache dictionary with the account data stored.

    This function is pure and does not mutate the provided cache.

    Args:
        cache (dict): Existing cache mapping.
        account_id (str): Unique identifier for the user.
        account_data (dict): Dictionary containing user details.

    Returns:
        dict: A new cache with the updated/inserted account entry.
    """
    new_cache = dict(cache)
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
    if not is_valid_account_info(account_id, account_data):
        return False

    try:
        global user_cache
        user_cache = store_account_info(user_cache, account_id, account_data)
        return True
    except Exception:
        # Any unexpected failure should return False per contract
        return False
