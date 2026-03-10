user_cache = {}

def validate_account_data(account_id, account_data):
    """Validate that account_id and account_data are appropriate for caching."""
    if not isinstance(account_id, str):
        return False
    if not isinstance(account_data, dict):
        return False
    if not account_id:
        return False
    return True

def store_in_cache(cache, account_id, account_data):
    """Store account data in the provided cache dictionary."""
    cache[account_id] = account_data
    return True

def cache_account_info(account_id, account_data):
    try:
        if not validate_account_data(account_id, account_data):
            return False
        return store_in_cache(user_cache, account_id, account_data)
    except:
        return False
