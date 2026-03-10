user_cache = {}

def retrieve_account_info(account_id: str):
    """
    Retrieve and return the entire account data for the specified account_id from the global in-memory cache.

    Args:
        account_id (str): The user's account identifier.

    Returns:
        The account data if present in the cache; otherwise, None.
    """
    return user_cache.get(account_id)

def cache_account_info(account_id: str, account_data: dict) -> bool:
    """
    Store account_data in the global user_cache under account_id.

    Args:
        account_id (str): The user's account identifier.
        account_data (dict): The user's account data to cache.

    Returns:
        bool: True if the data was successfully stored, False otherwise.
    """
    try:
        if not isinstance(account_id, str) or not account_id:
            return False
        if not isinstance(account_data, dict):
            return False

        user_cache[account_id] = account_data

        stored = user_cache.get(account_id, None)
        return stored is account_data or stored == account_data
    except Exception:
        return False
