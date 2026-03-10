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


def execute_transfer(identifier: str, role: str, source_acc: str, destination_acc: str, value: float) -> str:
    """
    Execute a funds transfer between accounts with role-based authorization.

    Args:
        identifier (str): The user ID of the requester.
        role (str): The role of the requester (e.g., 'account_owner', 'bank_admin').
        source_acc (str): The account number from which funds are withdrawn.
        destination_acc (str): The account number to which funds are deposited.
        value (float): The amount to transfer.

    Returns:
        str: A success message if the transfer completes.

    Raises:
        ValueError: If accounts are invalid, value is invalid, or insufficient funds.
        PermissionError: If the requester is not authorized to perform the transfer.
    """
    # Basic input validation
    if source_acc == destination_acc:
        raise ValueError("Source and destination accounts must be different")

    if source_acc not in ACCOUNTS_DB or destination_acc not in ACCOUNTS_DB:
        raise ValueError("One or both accounts not found")

    if not isinstance(value, (int, float)):
        raise ValueError("Transfer value must be a number")

    value = float(value)
    if value <= 0:
        raise ValueError("Transfer value must be greater than zero")

    source = ACCOUNTS_DB[source_acc]
    destination = ACCOUNTS_DB[destination_acc]

    # Authorization
    normalized_role = (role or "").strip().lower()
    if normalized_role == "bank_admin":
        authorized = True
    elif normalized_role == "account_owner":
        authorized = (identifier == source.get("owner_id"))
    else:
        authorized = False

    if not authorized:
        raise PermissionError("Not authorized to perform this transfer")

    # Funds availability
    if source.get("balance", 0.0) < value:
        raise ValueError("Insufficient funds")

    # Perform transfer
    source["balance"] = float(source.get("balance", 0.0)) - value
    destination["balance"] = float(destination.get("balance", 0.0)) + value

    return f"Transfer of {value} from {source_acc} to {destination_acc} completed successfully."
