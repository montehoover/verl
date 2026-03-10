user_cache = {}

def _is_valid_account_info(account_id: str, account_data: dict) -> bool:
    """
    Validates the account ID and account data.

    Args:
        account_id: The user's account ID.
        account_data: The user's account data.

    Returns:
        True if the account information is valid, False otherwise.
    """
    if not isinstance(account_id, str) or not account_id:
        return False  # account_id must be a non-empty string
    if not isinstance(account_data, dict):
        return False  # account_data must be a dictionary
    return True

def _store_account_info(cache: dict, account_id: str, account_data: dict) -> bool:
    """
    Stores the account data in the provided cache.

    Args:
        cache: The dictionary to store the account data in.
        account_id: The user's account ID.
        account_data: The user's account data.

    Returns:
        True if storage was successful, False otherwise.
    """
    try:
        cache[account_id] = account_data
        return True
    except Exception:
        # In a real-world scenario, log the exception here
        return False

def cache_account_info(account_id: str, account_data: dict) -> bool:
    """
    Caches user-specific data in a global dictionary.

    Args:
        account_id: A unique identifier for the user (string).
        account_data: A dictionary containing details about the user.

    Returns:
        True if the caching operation succeeds, False otherwise.
    """
    if not _is_valid_account_info(account_id, account_data):
        return False

    return _store_account_info(user_cache, account_id, account_data)
