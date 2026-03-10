"""
This module implements a simple in-memory caching system for user data.
It provides a function to store user-specific information in a global cache.
"""

user_cache = {}  # Global dictionary acting as the repository for cached user data.


def store_customer_info(cust_id: str, cust_data: dict) -> bool:
    """
    Implements a memory-based caching system for user data to optimize
    application speed. This function stores user-specific information in a
    dictionary, allowing for swift access without repeated database queries.

    Args:
        cust_id: A unique string that identifies each user.
        cust_data: A dictionary encompassing various user-related information.

    Returns:
        bool: Indicates True for successful caching, False for failure.
    """
    # Validate input types: cust_id must be a string and cust_data must be a dictionary.
    if not isinstance(cust_id, str):
        # Consider logging this failure or raising TypeError for stricter error handling.
        return False
    if not isinstance(cust_data, dict):
        # Consider logging this failure or raising TypeError for stricter error handling.
        return False
    
    # Attempt to store the customer data in the cache.
    try:
        # Add or update the customer's data in the user_cache dictionary.
        user_cache[cust_id] = cust_data
        return True  # Return True to indicate successful caching.
    except Exception:
        # Catch any exceptions that might occur during the dictionary assignment.
        # In a production environment, it's advisable to log this error
        # and potentially catch more specific exceptions.
        # Example: import logging; logging.exception(f"Error caching data for cust_id: {cust_id}")
        return False  # Return False to indicate a failure in caching.
