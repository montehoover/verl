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


def initiate_transfer(
    initiator_id: str,
    position: str,
    withdraw_account: str,
    deposit_account: str,
    transfer_value: float
) -> str:
    """
    Facilitates money transfers between two accounts within a banking system.

    Args:
        initiator_id: ID of the user initiating the transfer.
        position: Role of the user (e.g., 'account_owner', 'bank_admin', 'customer_service').
        withdraw_account: Account number from which funds will be withdrawn.
        deposit_account: Account number to which funds will be deposited.
        transfer_value: Total amount to be transferred.

    Returns:
        A success message confirming the transfer when authorized.

    Raises:
        PermissionError: If the initiator is not authorized to perform the transfer.
        ValueError: If accounts are invalid, amount is non-positive, accounts are the same, or insufficient funds.
    """
    if not isinstance(initiator_id, str) or not initiator_id:
        raise ValueError("initiator_id must be a non-empty string.")
    if not isinstance(position, str) or not position:
        raise ValueError("position must be a non-empty string.")
    if not isinstance(withdraw_account, str) or not withdraw_account:
        raise ValueError("withdraw_account must be a non-empty string.")
    if not isinstance(deposit_account, str) or not deposit_account:
        raise ValueError("deposit_account must be a non-empty string.")
    if not isinstance(transfer_value, (int, float)):
        raise ValueError("transfer_value must be a number.")
    transfer_amount = float(transfer_value)

    if transfer_amount <= 0:
        raise ValueError("transfer_value must be greater than 0.")
    if withdraw_account == deposit_account:
        raise ValueError("withdraw_account and deposit_account must be different.")

    withdraw_acct = ACCOUNTS_DB.get(withdraw_account)
    if withdraw_acct is None:
        raise ValueError(f"Withdraw account '{withdraw_account}' does not exist.")

    deposit_acct = ACCOUNTS_DB.get(deposit_account)
    if deposit_acct is None:
        raise ValueError(f"Deposit account '{deposit_account}' does not exist.")

    role = position.strip().lower()

    # Authorization rules:
    # - bank_admin: can transfer between any accounts
    # - account_owner: can transfer only from accounts they own
    # - others (e.g., customer_service): unauthorized by default
    is_authorized = False
    if role == "bank_admin":
        is_authorized = True
    elif role == "account_owner" and withdraw_acct.get("owner_id") == initiator_id:
        is_authorized = True

    if not is_authorized:
        raise PermissionError("Unauthorized transfer attempt.")

    # Funds availability check
    if withdraw_acct["balance"] < transfer_amount:
        raise ValueError("Insufficient funds in the withdraw account.")

    # Perform transfer
    withdraw_acct["balance"] -= transfer_amount
    deposit_acct["balance"] += transfer_amount

    return f"Successfully transferred ${transfer_amount} from {withdraw_account} to {deposit_account}"
