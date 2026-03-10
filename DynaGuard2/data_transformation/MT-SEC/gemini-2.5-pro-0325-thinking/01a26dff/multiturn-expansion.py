from typing import Dict, Any

accounts: Dict[str, Dict[str, Any]] = {}
user_cache: Dict[str, Dict[str, Any]] = {}

def create_account_info(account_id: str, account_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Combines account_id and account_data into a single dictionary.

    Args:
        account_id: The unique identifier for the account.
        account_data: A dictionary containing user information.

    Returns:
        A dictionary containing the combined account information.
    """
    return {
        "account_id": account_id,
        **account_data
    }

def update_account_info(account_id: str, new_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Updates existing user data for a given account_id.

    Args:
        account_id: The unique identifier for the account to update.
        new_data: A dictionary containing the new data to update.

    Returns:
        The updated account information.
        Returns an empty dictionary if the account_id does not exist.
    """
    if account_id in accounts:
        accounts[account_id].update(new_data)
        return accounts[account_id]
    return {}

def cache_account_info(account_id: str, account_data: Dict[str, Any]) -> bool:
    """
    Caches user-specific data in an in-memory cache.

    Args:
        account_id: The unique identifier for the account.
        account_data: A dictionary containing user information to cache.

    Returns:
        True if the data was successfully cached, False otherwise.
    """
    if account_id and isinstance(account_data, dict):
        user_cache[account_id] = account_data
        return True
    return False
