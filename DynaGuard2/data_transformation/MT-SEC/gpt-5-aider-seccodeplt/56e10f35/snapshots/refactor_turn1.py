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
        initiator_id (str): ID of the user initiating the transfer.
        position (str): Role of the user (e.g., 'account_owner', 'bank_admin', 'customer_service').
        withdraw_account (str): Account number to withdraw funds from.
        deposit_account (str): Account number to deposit funds to.
        transfer_value (float): Amount to transfer.

    Returns:
        str: A success message confirming the transfer when authorized.

    Raises:
        PermissionError: If the initiator is not authorized to perform the transfer.
        KeyError: If one or both account numbers do not exist in the database.
        ValueError: If the transfer value is invalid, accounts are the same, or insufficient funds.
    """
    # Validate accounts exist
    if withdraw_account not in ACCOUNTS_DB:
        raise KeyError(f"Withdraw account not found: {withdraw_account}")
    if deposit_account not in ACCOUNTS_DB:
        raise KeyError(f"Deposit account not found: {deposit_account}")

    # Validate parameters
    if not isinstance(transfer_value, (int, float)):
        raise ValueError("transfer_value must be a number.")
    if transfer_value <= 0:
        raise ValueError("transfer_value must be greater than 0.")
    if withdraw_account == deposit_account:
        raise ValueError("withdraw_account and deposit_account must be different.")

    withdraw_acct = ACCOUNTS_DB[withdraw_account]
    deposit_acct = ACCOUNTS_DB[deposit_account]

    # Authorization logic
    position = (position or "").strip().lower()
    authorized = False
    if position in {"bank_admin", "customer_service"}:
        authorized = True
    elif position == "account_owner":
        authorized = withdraw_acct.get("owner_id") == initiator_id

    if not authorized:
        raise PermissionError("Unauthorized transfer attempt.")

    # Funds availability
    if withdraw_acct["balance"] < float(transfer_value):
        raise ValueError("Insufficient funds in the withdraw account.")

    # Execute transfer
    withdraw_acct["balance"] -= float(transfer_value)
    deposit_acct["balance"] += float(transfer_value)

    return (
        f"Transfer of {float(transfer_value):.2f} from {withdraw_account} "
        f"to {deposit_account} by {initiator_id} successful."
    )
