# Global cache dictionary for storing user account information
user_cache = {}


def validate_account_data(account_id, account_data):
    """Validate that account_id and account_data are appropriate for caching.
    
    Args:
        account_id (str): Unique identifier for the user account
        account_data (dict): Dictionary containing user account details
        
    Returns:
        bool: True if validation passes, False otherwise
    """
    # Check if account_id is a string
    if not isinstance(account_id, str):
        return False
    
    # Check if account_data is a dictionary
    if not isinstance(account_data, dict):
        return False
    
    # Check if account_id is not empty
    if not account_id:
        return False
    
    return True


def store_in_cache(cache, account_id, account_data):
    """Store account data in the provided cache dictionary.
    
    Args:
        cache (dict): The cache dictionary to store data in
        account_id (str): Unique identifier for the user account
        account_data (dict): Dictionary containing user account details
        
    Returns:
        bool: Always returns True after successful storage
    """
    # Store the account data in the cache using account_id as key
    cache[account_id] = account_data
    return True


def cache_account_info(account_id, account_data):
    """Cache user account information for faster retrieval.
    
    This function validates the input data and stores it in the global
    user_cache dictionary to avoid frequent database queries.
    
    Args:
        account_id (str): Unique identifier for the user account
        account_data (dict): Dictionary containing user account details
        
    Returns:
        bool: True if caching succeeds, False if it fails
    """
    try:
        # Validate input data before caching
        if not validate_account_data(account_id, account_data):
            return False
        
        # Store validated data in the global cache
        return store_in_cache(user_cache, account_id, account_data)
    except:
        # Return False if any exception occurs during caching
        return False
