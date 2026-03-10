import logging

# Global cache dictionary for storing user data
user_cache = {}


def cache_application_data(client_id: str, info_data: dict) -> bool:
    """
    Store user-specific information in a memory-based cache.
    
    This function implements a simple caching mechanism to optimize 
    application performance by storing user data in memory, avoiding 
    repeated database queries.
    
    Args:
        client_id (str): A unique string that identifies each user.
        info_data (dict): A dictionary encompassing various user-related 
                         information to be cached.
    
    Returns:
        bool: True if the data was successfully cached, False if any 
              error occurred during the caching process.
    """
    # Initialize logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    try:
        # Store the user data in the global cache dictionary
        user_cache[client_id] = info_data
        
        # Log successful caching operation
        logger.info(f"Successfully cached data for client_id: {client_id}")
        
        # Return success status
        return True
        
    except Exception as e:
        # Log failure with error details
        logger.error(f"Failed to cache data for client_id: {client_id}. Error: {str(e)}")
        
        # Handle any unexpected errors during caching
        return False
