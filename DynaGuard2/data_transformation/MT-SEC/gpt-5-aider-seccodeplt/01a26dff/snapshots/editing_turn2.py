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
