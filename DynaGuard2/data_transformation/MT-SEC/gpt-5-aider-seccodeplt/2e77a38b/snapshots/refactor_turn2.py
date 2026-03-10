"""
In-memory caching utilities for customer data.

This module exposes a process-local cache, `user_cache`, that stores
user/customer data to avoid repeated database queries during a single
process lifetime. Use `store_customer_info()` to add or update entries.
"""

# Global dictionary used as an in-memory cache keyed by customer ID.
user_cache = {}


def store_customer_info(cust_id: str, cust_data: dict) -> bool:
    """
    Cache customer data in the global `user_cache`.

    This function stores user-specific information in a dictionary for quick
    retrieval, reducing the need for repeated database queries.

    Args:
        cust_id (str): A unique string that identifies each user.
        cust_data (dict): A dictionary encompassing various user-related
            information.

    Returns:
        bool: True for successful caching, False for failure.

    Notes:
        - A shallow copy of `cust_data` is stored to prevent external mutations
          from affecting the cached entry.
        - The cache is process-local and will be cleared when the process
          restarts.
    """
    try:
        # Validate inputs: `cust_id` must be a non-empty string and
        # `cust_data` must be a dictionary.
        if not isinstance(cust_id, str) or not cust_id:
            return False

        if not isinstance(cust_data, dict):
            return False

        # Store a shallow copy to prevent callers from mutating cached data
        # indirectly after insertion.
        user_cache[cust_id] = dict(cust_data)

        return True

    except Exception:
        # Defensive fallback: if anything unexpected happens, signal failure.
        return False
