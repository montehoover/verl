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


def shift_funds(
    initiator_id: str,
    initiator_role: str,
    withdraw_account: str,
    deposit_account: str,
    transfer_value: float
) -> str:
    """
    Move funds from one account to another with authorization checks.

    Parameters:
        initiator_id (str): The user ID initiating the transaction.
        initiator_role (str): Role of the user ('account_owner', 'bank_admin', 'customer_service').
        withdraw_account (str): The account number to withdraw funds from.
        deposit_account (str): The account number to deposit funds into.
        transfer_value (float): The amount of money to transfer (must be positive).

    Returns:
        str: Success message confirming the transfer.

    Raises:
        PermissionError: If the initiator is not authorized to perform the transfer.
        ValueError: If inputs are invalid, accounts are missing, or funds are insufficient.
    """
    # Basic validations
    if not isinstance(transfer_value, (int, float)):
        raise ValueError("transfer_value must be a number.")
    transfer_amount = float(transfer_value)
    if transfer_amount <= 0:
        raise ValueError("transfer_value must be positive.")
    if withdraw_account == deposit_account:
        raise ValueError("withdraw_account and deposit_account must be different.")

    # Account existence checks
    if withdraw_account not in ACCOUNTS_DB:
        raise ValueError(f"Withdraw account '{withdraw_account}' not found.")
    if deposit_account not in ACCOUNTS_DB:
        raise ValueError(f"Deposit account '{deposit_account}' not found.")

    withdraw_acc = ACCOUNTS_DB[withdraw_account]
    deposit_acc = ACCOUNTS_DB[deposit_account]

    # Authorization rules:
    # - bank_admin, customer_service: can transfer between any accounts
    # - account_owner: can withdraw only from an account they own (deposit to any)
    role = (initiator_role or "").lower()
    authorized = False
    if role in ("bank_admin", "customer_service"):
        authorized = True
    elif role == "account_owner" and withdraw_acc.get("owner_id") == initiator_id:
        authorized = True

    if not authorized:
        raise PermissionError("Unauthorized access: initiator is not permitted to perform this transfer.")

    # Funds check
    if withdraw_acc.get("balance", 0.0) < transfer_amount:
        raise ValueError("Insufficient funds in the withdraw account.")

    # Perform transfer
    withdraw_acc["balance"] -= transfer_amount
    deposit_acc["balance"] += transfer_amount

    return f"Transfer of {transfer_amount:.2f} from {withdraw_account} to {deposit_account} completed."
