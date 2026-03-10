"""
Banking application utilities.
Assumes ACCOUNTS_DB is a dictionary mapping account numbers (str)
to account records containing at least 'owner_id' and 'balance'.
"""

from typing import Any, Mapping
import math

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

def check_funds(account_number: str, amount: float) -> bool:
    """
    Verify whether the specified account has sufficient funds for a transaction.

    Args:
        account_number: The account number as a string.
        amount: The transaction amount as a non-negative float.

    Returns:
        True if the account's balance is greater than or equal to amount, otherwise False.

    Raises:
        ValueError: If account_number is not a non-empty string or amount is invalid.
        KeyError: If the account number does not exist in ACCOUNTS_DB.
    """
    if not isinstance(account_number, str) or not account_number.strip():
        raise ValueError("account_number must be a non-empty string")

    # Validate amount: must be a real, finite, non-negative number (bool not allowed)
    if isinstance(amount, bool) or not isinstance(amount, (int, float)):
        raise ValueError("amount must be a non-negative number")
    if not math.isfinite(float(amount)) or amount < 0:
        raise ValueError("amount must be a non-negative, finite number")

    details = get_account_details(account_number)
    balance = details.get("balance")

    try:
        return float(balance) >= float(amount)
    except (TypeError, ValueError):
        # If balance is missing or non-numeric, treat as insufficient funds
        return False
