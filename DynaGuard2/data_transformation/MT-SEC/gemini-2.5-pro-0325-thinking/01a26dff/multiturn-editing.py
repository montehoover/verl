from typing import Any, Optional

user_cache = {}

def retrieve_account_info(account_id: str) -> Optional[Any]:
    """
    Retrieves the account data for a specific user from the global in-memory cache.

    Args:
        account_id: The account ID of the user.

    Returns:
        The user's account data if found in the cache, otherwise None.
    """
    return user_cache.get(account_id)

def cache_account_info(account_id: str, account_data: dict) -> bool:
    """
    Stores user details into the global in-memory cache.

    Args:
        account_id: The account ID of the user.
        account_data: A dictionary containing the user's account details.

    Returns:
        True if the caching operation was successful, False otherwise.
        In this implementation, it will always return True as dictionary
        assignment itself doesn't return a status and exceptions are not caught.
    """
    user_cache[account_id] = account_data
    return True
