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


def process_transfer(
    uid: str,
    role: str,
    source_account: str,
    destination_account: str,
    transfer_amount: float
) -> str:
    """
    Facilitates a money transfer between two accounts.

    Parameters:
        uid (str): ID of the user initiating the transfer.
        role (str): Role of the user (e.g., 'account_owner', 'bank_admin', 'customer_service').
        source_account (str): Account number to withdraw funds from.
        destination_account (str): Account number to deposit funds to.
        transfer_amount (float): Amount to transfer.

    Returns:
        str: Success message confirming the transfer when authorized.

    Raises:
        PermissionError: If the user is not authorized to perform the transfer.
        ValueError: If inputs are invalid (e.g., accounts missing, insufficient funds, invalid amount).
    """
    # Validate accounts exist
    if source_account not in ACCOUNTS_DB:
        raise ValueError(f"Source account not found: {source_account}")
    if destination_account not in ACCOUNTS_DB:
        raise ValueError(f"Destination account not found: {destination_account}")

    src = ACCOUNTS_DB[source_account]
    dst = ACCOUNTS_DB[destination_account]

    # Validate transfer amount
    if not isinstance(transfer_amount, (int, float)):
        raise ValueError("transfer_amount must be a number")
    if transfer_amount <= 0:
        raise ValueError("transfer_amount must be greater than 0")

    # Prevent same-account transfer
    if source_account == destination_account:
        raise ValueError("source_account and destination_account must be different")

    # Authorization checks
    authorized = False
    if role == "bank_admin":
        authorized = True
    elif role == "account_owner":
        authorized = (src.get("owner_id") == uid)
    elif role == "customer_service":
        authorized = False  # Not authorized to execute transfers
    else:
        authorized = False  # Unknown roles are not authorized

    if not authorized:
        raise PermissionError("Unauthorized transfer attempt")

    # Funds check
    if src.get("balance", 0.0) < transfer_amount:
        raise ValueError("Insufficient funds")

    # Perform transfer
    src["balance"] -= transfer_amount
    dst["balance"] += transfer_amount

    return f"Successfully transferred ${transfer_amount} from {source_account} to {destination_account}"
