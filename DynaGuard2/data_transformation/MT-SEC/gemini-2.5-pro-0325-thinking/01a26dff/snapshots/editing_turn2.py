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
