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


def process_transfer(uid: str, role: str, source_account: str, destination_account: str, transfer_amount: float) -> str:
    """
    Facilitates money transfers between two accounts within a banking system.

    Args:
        uid: ID of the user initiating the transfer.
        role: User's role (e.g., 'account_owner', 'bank_admin', 'customer_service').
        source_account: Account number to withdraw funds from.
        destination_account: Account number to deposit funds to.
        transfer_amount: Amount to transfer.

    Returns:
        Success message confirming the transfer when authorized.

    Raises:
        PermissionError: If the user is not authorized to perform the transfer.
        ValueError: If input is invalid, accounts do not exist, or insufficient funds.
    """
    if not isinstance(uid, str) or not uid:
        raise ValueError("Invalid uid.")
    if not isinstance(role, str) or not role:
        raise ValueError("Invalid role.")
    if not isinstance(source_account, str) or not source_account:
        raise ValueError("Invalid source_account.")
    if not isinstance(destination_account, str) or not destination_account:
        raise ValueError("Invalid destination_account.")
    if not isinstance(transfer_amount, (int, float)):
        raise ValueError("transfer_amount must be a number.")
    if transfer_amount <= 0:
        raise ValueError("transfer_amount must be greater than 0.")
    if source_account == destination_account:
        raise ValueError("source_account and destination_account must be different.")

    role_normalized = role.strip().lower()

    # Authorization:
    # - bank_admin: can transfer between any accounts.
    # - account_owner: can transfer from accounts they own.
    # - others (e.g., customer_service): not authorized to execute transfers.
    if role_normalized == "bank_admin":
        authorized = True
    elif role_normalized == "account_owner":
        src = ACCOUNTS_DB.get(source_account)
        if src is None:
            raise ValueError(f"Source account '{source_account}' does not exist.")
        authorized = (src.get("owner_id") == uid)
    else:
        authorized = False

    if not authorized:
        raise PermissionError("Unauthorized transfer attempt.")

    # Account existence checks
    src_acct = ACCOUNTS_DB.get(source_account)
    if src_acct is None:
        raise ValueError(f"Source account '{source_account}' does not exist.")
    dst_acct = ACCOUNTS_DB.get(destination_account)
    if dst_acct is None:
        raise ValueError(f"Destination account '{destination_account}' does not exist.")

    # Funds availability
    current_balance = float(src_acct.get("balance", 0.0))
    if current_balance < transfer_amount:
        raise ValueError("Insufficient funds in the source account.")

    # Perform transfer
    src_acct["balance"] = current_balance - transfer_amount
    dst_acct["balance"] = float(dst_acct.get("balance", 0.0)) + transfer_amount

    return (
        f"Transfer of {transfer_amount:.2f} from {source_account} to "
        f"{destination_account} completed."
    )
