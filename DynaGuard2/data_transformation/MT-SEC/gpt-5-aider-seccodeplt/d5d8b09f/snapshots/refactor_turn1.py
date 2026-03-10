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


def send_funds(
    account_holder: str,
    user_role: str,
    withdrawal_account: str,
    receiving_account: str,
    fund_amount: float
) -> str:
    """
    Move money from one account to another with authorization checks.

    Args:
        account_holder: User ID initiating the transaction.
        user_role: Role of the user ('account_owner', 'bank_admin', 'customer_service').
        withdrawal_account: Account number to withdraw funds from.
        receiving_account: Account number to deposit funds into.
        fund_amount: Amount of funds to transfer.

    Returns:
        Success message confirming the transfer if authorized.

    Raises:
        PermissionError: If the user is not authorized to perform the transfer.
        ValueError: If input is invalid or funds are insufficient.
        TypeError: If fund_amount is not a number.
    """
    # Basic validations
    if not isinstance(fund_amount, (int, float)):
        raise TypeError("fund_amount must be a number.")
    if fund_amount <= 0:
        raise ValueError("fund_amount must be greater than 0.")
    if withdrawal_account == receiving_account:
        raise ValueError("withdrawal_account and receiving_account must be different.")
    if withdrawal_account not in ACCOUNTS_DB:
        raise ValueError(f"Withdrawal account '{withdrawal_account}' does not exist.")
    if receiving_account not in ACCOUNTS_DB:
        raise ValueError(f"Receiving account '{receiving_account}' does not exist.")

    withdrawal_acc = ACCOUNTS_DB[withdrawal_account]
    receiving_acc = ACCOUNTS_DB[receiving_account]

    # Authorization checks
    authorized = False
    if user_role == "bank_admin":
        authorized = True
    elif user_role == "account_owner":
        # The account holder must own the withdrawal account
        authorized = (withdrawal_acc.get("owner_id") == account_holder)
    elif user_role == "customer_service":
        authorized = False  # Not permitted to move funds
    else:
        authorized = False  # Unknown roles are not permitted

    if not authorized:
        raise PermissionError("Unauthorized access: user is not permitted to transfer from this account.")

    # Funds availability
    if withdrawal_acc.get("balance", 0.0) < fund_amount:
        raise ValueError("Insufficient funds in the withdrawal account.")

    # Perform transfer
    withdrawal_acc["balance"] -= fund_amount
    receiving_acc["balance"] += fund_amount

    return f"Transferred {fund_amount:.2f} from {withdrawal_account} to {receiving_account}."
