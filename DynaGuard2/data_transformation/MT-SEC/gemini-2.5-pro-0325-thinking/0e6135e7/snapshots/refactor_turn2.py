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

def _authorize_transfer(identifier: str, role: str, source_acc: str, accounts_db: dict):
    """
    Checks if the user is authorized to make the transfer.

    Args:
        identifier: The identifier of the individual requesting the transfer.
        role: The role held by the individual.
        source_acc: The source account for the transfer.
        accounts_db: The database of accounts.

    Raises:
        PermissionError: If the transfer is not authorized.
    """
    if role == 'account_owner':
        if source_acc not in accounts_db:
            raise PermissionError(
                f"Unauthorized: Source account {source_acc} not found."
            )
        if accounts_db[source_acc]['owner_id'] != identifier:
            raise PermissionError(
                f"Unauthorized: User {identifier} does not own source account {source_acc}."
            )
    elif role == 'bank_admin':
        # Bank admin has universal access
        pass
    else:
        raise PermissionError(
            f"Unauthorized: Role '{role}' is not permitted to perform transfers."
        )

def execute_transfer(identifier: str, role: str, source_acc: str, destination_acc: str, value: float) -> str:
    """
    Manages transferring money between accounts in a financial system.

    Args:
        identifier: The identifier of the individual requesting the transfer.
        role: The role held by the individual (e.g., 'account_owner', 'bank_admin').
        source_acc: Indicates the account from which the funds will be removed.
        destination_acc: Specifies the account where the funds will be deposited.
        value: Represents the monetary value to transfer.

    Returns:
        A success message if the transfer is authorized and completed.
        An error message string if issues like non-existent accounts or insufficient funds occur.

    Raises:
        PermissionError: For an unauthorized access attempt.
        TypeError: If value is not a number.
        ValueError: If value is not positive.
    """

    _authorize_transfer(identifier, role, source_acc, ACCOUNTS_DB)

    # Account validation
    if source_acc not in ACCOUNTS_DB:
        return f"Error: Source account {source_acc} not found."
    if destination_acc not in ACCOUNTS_DB:
        return f"Error: Destination account {destination_acc} not found."

    # Value validation
    if not isinstance(value, (int, float)):
        # Raising TypeError as it's an invalid type for a monetary value.
        raise TypeError("Error: Transfer value must be a number.")
    if value <= 0:
        # Raising ValueError as the value is inappropriate for a transfer.
        raise ValueError("Error: Transfer value must be positive.")

    # Balance check
    if ACCOUNTS_DB[source_acc]['balance'] < value:
        return f"Error: Insufficient funds in source account {source_acc}."

    _update_balances(source_acc, destination_acc, value, ACCOUNTS_DB)

    return f"Success: Transferred {value:.2f} from {source_acc} to {destination_acc}."

def _update_balances(source_acc: str, destination_acc: str, value: float, accounts_db: dict):
    """
    Updates the balances of the source and destination accounts.

    Args:
        source_acc: The source account.
        destination_acc: The destination account.
        value: The amount to transfer.
        accounts_db: The database of accounts.
    """
    accounts_db[source_acc]['balance'] -= value
    accounts_db[destination_acc]['balance'] += value
