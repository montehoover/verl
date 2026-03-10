"""
Banking application utilities.
Assumes ACCOUNTS_DB is a dictionary mapping account numbers (str)
to account records containing at least 'owner_id' and 'balance'.
"""

from typing import Any, MutableMapping
import math

# Placeholder database; can be overridden by the application environment.
ACCOUNTS_DB: MutableMapping[str, Any] = {}

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

def send_funds(
    account_holder: str,
    user_role: str,
    withdrawal_account: str,
    receiving_account: str,
    fund_amount: float
) -> str:
    """
    Transfer funds between accounts.

    Args:
        account_holder: The user ID initiating the transaction.
        user_role: The role of the user (e.g., 'admin', 'user').
        withdrawal_account: Account number to withdraw funds from.
        receiving_account: Account number to deposit funds into.
        fund_amount: Amount to transfer (must be positive).

    Returns:
        A success message if the transfer is confirmed. If the user is unauthorized,
        a PermissionError is raised.

    Raises:
        ValueError: If inputs are invalid or balances are malformed.
        KeyError: If either account does not exist in ACCOUNTS_DB.
        PermissionError: If the user is not authorized to transfer from the withdrawal account.
    """
    # Validate string inputs
    if not isinstance(account_holder, str) or not account_holder.strip():
        raise ValueError("account_holder must be a non-empty string")
    if not isinstance(user_role, str) or not user_role.strip():
        raise ValueError("user_role must be a non-empty string")
    if not isinstance(withdrawal_account, str) or not withdrawal_account.strip():
        raise ValueError("withdrawal_account must be a non-empty string")
    if not isinstance(receiving_account, str) or not receiving_account.strip():
        raise ValueError("receiving_account must be a non-empty string")

    # Validate amount: positive, finite number (bool not allowed)
    if isinstance(fund_amount, bool) or not isinstance(fund_amount, (int, float)):
        raise ValueError("fund_amount must be a positive number")
    fund_amount = float(fund_amount)
    if not math.isfinite(fund_amount) or fund_amount <= 0:
        raise ValueError("fund_amount must be a positive, finite number")

    # Retrieve accounts (raises KeyError if not found)
    withdraw_details = get_account_details(withdrawal_account)
    receive_details = get_account_details(receiving_account)

    # Authorization: owner of withdrawal account or admin role
    role = user_role.strip().lower()
    is_authorized = (account_holder == withdraw_details.get("owner_id")) or (role == "admin")
    if not is_authorized:
        raise PermissionError("Unauthorized access")

    # Sufficient funds check
    if not check_funds(withdrawal_account, fund_amount):
        return "Insufficient funds for transfer"

    # Perform transfer
    try:
        withdraw_balance = float(ACCOUNTS_DB[withdrawal_account]["balance"])
        receive_balance = float(ACCOUNTS_DB[receiving_account]["balance"])
    except (KeyError, TypeError, ValueError):
        raise ValueError("Invalid account balance data")

    ACCOUNTS_DB[withdrawal_account]["balance"] = withdraw_balance - fund_amount
    ACCOUNTS_DB[receiving_account]["balance"] = receive_balance + fund_amount

    return f"Transfer of {fund_amount} from {withdrawal_account} to {receiving_account} confirmed"
