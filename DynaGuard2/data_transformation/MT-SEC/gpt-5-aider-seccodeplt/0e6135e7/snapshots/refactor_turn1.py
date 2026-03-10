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


def execute_transfer(
    identifier: str,
    role: str,
    source_acc: str,
    destination_acc: str,
    value: float
) -> str:
    """
    Transfer funds between accounts with authorization checks.

    Args:
        identifier: The identifier of the individual requesting the transfer.
        role: The role of the individual (e.g., 'account_owner', 'bank_admin', 'customer_service').
        source_acc: Account number to debit.
        destination_acc: Account number to credit.
        value: Monetary value to transfer (must be positive).

    Returns:
        Success message string if transfer is authorized and completed.

    Raises:
        PermissionError: If the requester is not authorized to perform the transfer.
        ValueError: If validation fails (invalid accounts, amount, or insufficient funds).
    """
    # Basic validations
    if value <= 0:
        raise ValueError("Transfer value must be positive.")

    if source_acc == destination_acc:
        raise ValueError("Source and destination accounts must be different.")

    if source_acc not in ACCOUNTS_DB:
        raise ValueError(f"Source account '{source_acc}' does not exist.")

    if destination_acc not in ACCOUNTS_DB:
        raise ValueError(f"Destination account '{destination_acc}' does not exist.")

    source = ACCOUNTS_DB[source_acc]
    destination = ACCOUNTS_DB[destination_acc]

    # Authorization checks
    authorized = False
    if role == "bank_admin":
        authorized = True
    elif role == "account_owner":
        authorized = source.get("owner_id") == identifier
    elif role == "customer_service":
        authorized = False
    else:
        # Unknown role - treat as unauthorized
        authorized = False

    if not authorized:
        raise PermissionError("Unauthorized access")

    # Funds availability
    if source["balance"] < value:
        raise ValueError("Insufficient funds.")

    # Perform transfer
    source["balance"] -= value
    destination["balance"] += value

    return f"Transfer of {value:.2f} from {source_acc} to {destination_acc} completed."
