user_cache = {}

def check_account_info(account_id: str) -> bool:
    """
    Checks if a specific user's data is stored in the global in-memory cache.

    Args:
        account_id: The account ID of the user.

    Returns:
        True if the user's info is cached, False otherwise.
    """
    return account_id in user_cache
