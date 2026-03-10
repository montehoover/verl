user_cache = {}

def check_account_info(account_id: str) -> bool:
    """
    Determine if a specific user's data is stored in the global in-memory cache.

    Args:
        account_id (str): The user's account identifier.

    Returns:
        bool: True if the user's info is cached, False otherwise.
    """
    return account_id in user_cache
