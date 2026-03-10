"""
Module providing a simple user cache and a function to store account info.

The cache is a global dictionary `user_cache` used to store user-specific data
for rapid retrieval, reducing repeated database queries.
"""

# Global cache storage
user_cache = {}

def cache_account_info(account_id: str, account_data: dict) -> bool:
    """
    Cache user-specific data keyed by a unique account identifier.

    Args:
        account_id (str): Unique identifier for the user.
        account_data (dict): Dictionary containing user details to cache.

    Returns:
        bool: True if caching succeeds, False otherwise.
    """
    # Validate inputs
    if not isinstance(account_id, str) or not account_id:
        return False
    if not isinstance(account_data, dict):
        return False

    try:
        # Store or update the cache entry
        user_cache[account_id] = account_data
        return True
    except Exception:
        # Any unexpected failure should return False per contract
        return False
