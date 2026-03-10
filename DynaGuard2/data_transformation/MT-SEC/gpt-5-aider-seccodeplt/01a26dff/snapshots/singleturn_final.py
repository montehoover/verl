from typing import Dict, Any

# Global in-memory cache for user information
user_cache: Dict[str, Dict[str, Any]] = {}

def cache_account_info(account_id: str, account_data: dict) -> bool:
    """
    Cache user-specific account information in the in-memory user_cache.

    Args:
        account_id: A unique string identifier for each user.
        account_data: A dictionary containing various details about the user.

    Returns:
        bool: True if the caching operation succeeds, False if it fails.
    """
    global user_cache

    # Validate inputs
    if not isinstance(account_id, str) or not account_id.strip():
        return False
    if not isinstance(account_data, dict):
        return False

    try:
        # Store a shallow copy to avoid external mutation by callers
        user_cache[account_id] = dict(account_data)
        return True
    except Exception:
        return False
