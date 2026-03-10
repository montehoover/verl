user_cache = {}

def cache_account_info(account_id: str, account_data: dict) -> bool:
    """
    Caches user-specific data in a global dictionary.

    Args:
        account_id: A unique identifier for the user (string).
        account_data: A dictionary containing details about the user.

    Returns:
        True if the caching operation succeeds, False otherwise.
    """
    if not isinstance(account_id, str) or not account_id:
        return False  # account_id must be a non-empty string
    if not isinstance(account_data, dict):
        return False  # account_data must be a dictionary

    try:
        user_cache[account_id] = account_data
        return True
    except Exception:
        # In a real-world scenario, log the exception here
        return False
