# Global cache for storing user profile data.
# Keys are profile_ids (str), values are profile_data (dict).
user_cache = {}


def cache_profile_data(profile_id: str, profile_data: dict) -> bool:
    """
    Caches user-specific data in a dictionary for rapid retrieval.

    This function first validates the types of the input parameters.
    If valid, it stores the `profile_data` associated with a `profile_id`
    in the global `user_cache`.

    Args:
        profile_id: A unique string identifier for each user. Must be a string.
        profile_data: A dictionary containing various details about the user.
                      Must be a dictionary.

    Returns:
        True if the caching operation was successful.
        False if the operation failed, for instance, due to invalid input
        types or an internal error during the caching process.
    """
    # Validate that profile_id is a string.
    if not isinstance(profile_id, str):
        # For production systems, consider using the logging module for errors.
        # e.g., import logging; logging.warning(f"Invalid profile_id type: {type(profile_id)}")
        return False

    # Validate that profile_data is a dictionary.
    if not isinstance(profile_data, dict):
        # For production systems, consider using the logging module for errors.
        # e.g., import logging; logging.warning(f"Invalid profile_data type: {type(profile_data)}")
        return False

    # Attempt to store the profile data in the cache.
    try:
        user_cache[profile_id] = profile_data
        return True
    except Exception as e:
        # An error occurred during the caching process.
        # For production systems, consider using the logging module for errors.
        # e.g., import logging; logging.error(f"Failed to cache data for {profile_id}: {e}", exc_info=True)
        return False
