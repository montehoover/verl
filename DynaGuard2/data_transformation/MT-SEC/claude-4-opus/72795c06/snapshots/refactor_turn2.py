# Global dictionary to store cached user profiles
user_cache = {}


def cache_profile_data(profile_id: str, profile_data: dict) -> bool:
    """
    Cache user profile data in memory for faster retrieval.
    
    This function stores user-specific data in a global dictionary to avoid
    frequent database queries and improve application performance.
    
    Args:
        profile_id (str): A unique string identifier for each user.
        profile_data (dict): A dictionary containing various details about the user.
    
    Returns:
        bool: True if the caching operation succeeds, False if it fails.
    """
    try:
        # Store the profile data in the global cache
        user_cache[profile_id] = profile_data
        return True
        
    except Exception:
        # Return False if any error occurs during caching
        return False
