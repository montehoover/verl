"""
Banking application utilities.
Assumes ACCOUNTS_DB is a dictionary mapping account numbers (str)
to account records containing at least 'owner_id' and 'balance'.
"""

from typing import Any, Mapping

# Placeholder database; can be overridden by the application environment.
ACCOUNTS_DB: Mapping[str, Any] = {}

def get_account_details(account_number: str) -> dict:
    """
    Retrieve details for a specific account.

    Args:
        account_number: The account number as a string.

    Returns:
        A dictionary containing at least 'owner_id' and 'balance' for the account.

    Raises:
        ValueError: If account_number is not a non-empty string.
        KeyError: If the account number does not exist in ACCOUNTS_DB.
    """
    if not isinstance(account_number, str) or not account_number.strip():
        raise ValueError("account_number must be a non-empty string")

    try:
        account = ACCOUNTS_DB[account_number]
    except KeyError as e:
        raise KeyError(f"Account not found: {account_number}") from e

    # Extract required fields; assume dict-like structure
    if not isinstance(account, dict):
        # If the structure is unexpected, wrap what we can safely return
        return {
            "owner_id": getattr(account, "owner_id", None),
            "balance": getattr(account, "balance", None),
        }

    return {
        "owner_id": account.get("owner_id"),
        "balance": account.get("balance"),
    }
