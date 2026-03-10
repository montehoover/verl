import logging

user_cache = {}  # Global dictionary acting as the repository for cached user data.


# Configure logging
# It's generally better to configure logging once at the application's entry point.
# However, per the request, initializing basicConfig here for demonstration.
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def cache_application_data(client_id: str, info_data: dict) -> bool:
    """
    Implements a memory-based caching system for user data.

    This function stores user-specific information in a dictionary, 
    allowing for swift access without repeated database queries.

    Args:
        client_id: A unique string that identifies each user.
        info_data: A dictionary encompassing various user-related information.

    Returns:
        True for successful caching, False for failure (e.g., incorrect input types).
    """
    # Validate input types:
    # - client_id must be a string.
    # - info_data must be a dictionary.
    # If type validation fails, return False immediately.
    if not isinstance(client_id, str):
        logging.error(f"Cache add failed for client_id '{client_id}': Invalid client_id type. Expected str, got {type(client_id).__name__}.")
        return False
    if not isinstance(info_data, dict):
        logging.error(f"Cache add failed for client_id '{client_id}': Invalid info_data type. Expected dict, got {type(info_data).__name__}.")
        return False
    
    try:
        # Perform the caching operation by assigning info_data to the client_id key in user_cache.
        user_cache[client_id] = info_data
        logging.info(f"Data cached successfully for client_id: '{client_id}'.")
        return True
    except Exception as e:
        logging.error(f"Cache add failed for client_id '{client_id}': An exception occurred: {e}")
        return False
