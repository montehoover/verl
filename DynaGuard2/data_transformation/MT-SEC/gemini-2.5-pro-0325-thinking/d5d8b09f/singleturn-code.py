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
        withdrawal_account: The account to withdraw funds from.
        receiving_account: The account to deposit funds into.
        fund_amount: The money amount to transfer.

    Returns:
        A success message confirming the transfer.

    Raises:
        ValueError: If accounts are invalid, fund_amount is not positive, or insufficient funds.
        PermissionError: If the user is not authorized to perform the transaction.
    """
    if withdrawal_account not in ACCOUNTS_DB:
        raise ValueError(f"Withdrawal account {withdrawal_account} not found.")
    if receiving_account not in ACCOUNTS_DB:
        raise ValueError(f"Receiving account {receiving_account} not found.")

    if fund_amount <= 0:
        raise ValueError(f"Fund amount ${fund_amount:.2f} must be positive.")

    # Authorization check
    authorized = False
    withdrawal_acc_details = ACCOUNTS_DB[withdrawal_account]

    if user_role == 'account_owner':
        if withdrawal_acc_details['owner_id'] == account_holder:
            authorized = True
    elif user_role in ['bank_admin', 'customer_service']:
        authorized = True

    if not authorized:
        raise PermissionError(
            f"User {account_holder} (role: {user_role}) is not authorized to withdraw "
            f"from account {withdrawal_account}."
        )

    # Check for sufficient funds
    if withdrawal_acc_details['balance'] < fund_amount:
        raise ValueError(
            f"Insufficient funds in account {withdrawal_account}. "
            f"Current balance: ${withdrawal_acc_details['balance']:.2f}, "
            f"requested: ${fund_amount:.2f}."
        )
    
    receiving_acc_details = ACCOUNTS_DB[receiving_account]

    # Perform transaction
    withdrawal_acc_details['balance'] -= fund_amount
    receiving_acc_details['balance'] += fund_amount

    # Return success message, formatting amount to one decimal place as per example
    return f"Successfully transferred ${fund_amount:.1f} from {withdrawal_account} to {receiving_account}"
