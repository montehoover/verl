"""
This module implements a simple in-memory caching system for user data.
It provides a function to store user-specific information in a global cache.
"""
import logging

# Configure basic logging for the module. 
# This is a common place to set up logging, 
# but the request specified initialization within the function.
# For a real application, this would typically be done once at startup.
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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
    # Initialize logger within the function as requested.
    # For broader applications, logger is often obtained at module level:
    # logger = logging.getLogger(__name__)
    logger = logging.getLogger() # Get the root logger configured by basicConfig

    # Validate input types: cust_id must be a string and cust_data must be a dictionary.
    if not isinstance(cust_id, str):
        logger.error(f"Invalid cust_id type: {type(cust_id)}. Expected str. cust_id: {cust_id}")
        return False
    if not isinstance(cust_data, dict):
        logger.error(f"Invalid cust_data type: {type(cust_data)}. Expected dict. cust_id: {cust_id}")
        return False
    
    # Attempt to store the customer data in the cache.
    try:
        # Add or update the customer's data in the user_cache dictionary.
        user_cache[cust_id] = cust_data
        logger.info(f"Successfully cached data for cust_id: {cust_id}")
        return True  # Return True to indicate successful caching.
    except Exception as e:
        # Catch any exceptions that might occur during the dictionary assignment.
        logger.exception(f"Error caching data for cust_id: {cust_id}. Exception: {e}")
        return False  # Return False to indicate a failure in caching.
