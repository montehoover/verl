from typing import Dict, Any

# In-memory storage for account data
accounts: Dict[str, Dict[str, Any]] = {}

def create_account_info(account_id: str, account_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create a combined account info dictionary.

    Args:
        account_id: Unique identifier for the account.
        account_data: Dictionary of user information.

    Returns:
        A dictionary combining the account_id and the provided account_data.
        If 'account_id' exists in account_data, the explicit account_id parameter takes precedence.
    """
    if not isinstance(account_id, str):
        raise TypeError("account_id must be a string")
    if not isinstance(account_data, dict):
        raise TypeError("account_data must be a dictionary")

    # Make a shallow copy to avoid mutating the caller's dictionary
    combined: Dict[str, Any] = dict(account_data)
    # Ensure account_id key is present and consistent
    combined["account_id"] = account_id
    return combined

def update_account_info(account_id: str, new_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Update existing user data in the in-memory accounts store.

    Args:
        account_id: Unique identifier for the account to update.
        new_data: Dictionary of fields to update.

    Returns:
        The updated account information dictionary.

    Raises:
        TypeError: If account_id is not a string or new_data is not a dictionary.
        KeyError: If the account_id does not exist in the accounts store.
    """
    if not isinstance(account_id, str):
        raise TypeError("account_id must be a string")
    if not isinstance(new_data, dict):
        raise TypeError("new_data must be a dictionary")

    if account_id not in accounts:
        raise KeyError(f"Account '{account_id}' does not exist")

    # Merge updates into a copy to avoid mutating any external references
    updated: Dict[str, Any] = dict(accounts[account_id])
    updated.update(new_data)
    # Ensure the account_id remains authoritative
    updated["account_id"] = account_id

    accounts[account_id] = updated
    return updated
