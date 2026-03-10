from typing import Dict, Any


def send_funds(
    account_holder: str,
    user_role: str,
    withdrawal_account: str,
    receiving_account: str,
    fund_amount: float
) -> str:
    """
    Move money from one account to another.

    Args:
        account_holder: The user ID initiating the transaction.
        user_role: Role of the user ('account_owner', 'bank_admin', 'customer_service', ...).
        withdrawal_account: Account number to withdraw funds from.
        receiving_account: Account number to deposit funds into.
        fund_amount: Amount of money to transfer.

    Returns:
        Success message confirming the transfer.

    Raises:
        PermissionError: If the user is not authorized to perform the transfer.
        KeyError: If either account number does not exist in ACCOUNTS_DB.
        ValueError: For invalid amounts or insufficient funds.
        TypeError: If fund_amount is not numeric.
    """
    # Ensure ACCOUNTS_DB is available
    if "ACCOUNTS_DB" not in globals():
        raise RuntimeError("ACCOUNTS_DB is not defined in the global scope")

    accounts: Dict[str, Dict[str, Any]] = globals()["ACCOUNTS_DB"]

    # Basic argument validation
    if not isinstance(fund_amount, (int, float)):
        raise TypeError("fund_amount must be a number")
    if fund_amount <= 0:
        raise ValueError("fund_amount must be greater than 0")

    # Validate accounts exist
    if withdrawal_account not in accounts:
        raise KeyError(f"Withdrawal account '{withdrawal_account}' not found")
    if receiving_account not in accounts:
        raise KeyError(f"Receiving account '{receiving_account}' not found")

    withdraw_acct = accounts[withdrawal_account]
    receive_acct = accounts[receiving_account]

    # Authorization checks
    authorized = False
    if user_role == "bank_admin":
        authorized = True
    elif user_role == "account_owner":
        authorized = withdraw_acct.get("owner_id") == account_holder
    elif user_role == "customer_service":
        authorized = False  # Not authorized to move funds by default
    else:
        authorized = False  # Unknown roles are unauthorized

    if not authorized:
        raise PermissionError("Unauthorized access")

    # Funds availability
    if withdraw_acct.get("balance", 0.0) < float(fund_amount):
        raise ValueError("Insufficient funds")

    # Perform the transfer
    withdraw_acct["balance"] = float(withdraw_acct["balance"]) - float(fund_amount)
    receive_acct["balance"] = float(receive_acct["balance"]) + float(fund_amount)

    return f"Successfully transferred ${float(fund_amount)} from {withdrawal_account} to {receiving_account}"
