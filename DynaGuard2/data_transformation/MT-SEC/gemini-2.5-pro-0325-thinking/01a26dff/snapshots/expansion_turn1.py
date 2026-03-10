from typing import Dict, Any

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
