ACCOUNTS_DB = {
    "ACC001": {
        "account_number": "ACC001",
        "owner_id": "USER1",
        "balance": 1000.0
    },
    "ACC002": {
        "account_number": "ACC002",
        "owner_id": "USER2",
        "balance": 500.0
    }
}

def send_funds(account_holder: str, user_role: str, withdrawal_account: str, receiving_account: str, fund_amount: float) -> str:
    """
    Handles moving money from one account to another in a banking system.

    Args:
        account_holder: The user ID initiating the transaction.
        user_role: The role of the user (e.g., 'account_owner', 'bank_admin', 'customer_service').
        withdrawal_account: The account ID to withdraw funds from.
        receiving_account: The account ID to deposit funds into.
        fund_amount: The amount of money to transfer.

    Returns:
        A success message confirming the transfer if the user is authorized and the transaction is valid.

    Raises:
        PermissionError: If the user is not authorized to perform the transaction.
        ValueError: If the fund amount is invalid, accounts are not found, or insufficient funds.
    """

    # Authorization check
    authorized = False
    if user_role in ['bank_admin', 'customer_service']:
        authorized = True
    elif user_role == 'account_owner':
        if withdrawal_account in ACCOUNTS_DB and ACCOUNTS_DB[withdrawal_account]['owner_id'] == account_holder:
            authorized = True
    
    if not authorized:
        raise PermissionError(
            f"User '{account_holder}' with role '{user_role}' is not authorized to initiate a transfer from account '{withdrawal_account}'."
        )

    # Validate fund amount
    if not isinstance(fund_amount, (int, float)) or fund_amount <= 0:
        raise ValueError("Fund amount must be a positive number.")

    # Validate accounts
    if withdrawal_account not in ACCOUNTS_DB:
        raise ValueError(f"Withdrawal account '{withdrawal_account}' not found.")
    if receiving_account not in ACCOUNTS_DB:
        raise ValueError(f"Receiving account '{receiving_account}' not found.")
    
    if withdrawal_account == receiving_account:
        raise ValueError("Withdrawal and receiving accounts cannot be the same.")

    # Check for sufficient funds
    if ACCOUNTS_DB[withdrawal_account]['balance'] < fund_amount:
        raise ValueError(
            f"Insufficient funds in account '{withdrawal_account}'. Current balance: {ACCOUNTS_DB[withdrawal_account]['balance']}, requested: {fund_amount}."
        )

    # Perform the transfer
    ACCOUNTS_DB[withdrawal_account]['balance'] -= fund_amount
    ACCOUNTS_DB[receiving_account]['balance'] += fund_amount

    return (
        f"Successfully transferred {fund_amount} from account '{withdrawal_account}' to account '{receiving_account}'. "
        f"Initiated by user '{account_holder}' ({user_role})."
    )
