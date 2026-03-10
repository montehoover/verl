from typing import Dict, Any

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
