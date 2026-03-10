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

    # Authorization
    if role == 'account_owner':
        if source_acc not in ACCOUNTS_DB:
            raise PermissionError(
                f"Unauthorized: Source account {source_acc} not found."
            )
        if ACCOUNTS_DB[source_acc]['owner_id'] != identifier:
            raise PermissionError(
                f"Unauthorized: User {identifier} does not own source account {source_acc}."
            )
    elif role == 'bank_admin':
        # Bank admin has universal access to initiate transfers
        pass
    else:
        # Other roles are not authorized for transfer
        raise PermissionError(
            f"Unauthorized: Role '{role}' is not permitted to perform transfers."
        )

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

    # Perform transfer
    ACCOUNTS_DB[source_acc]['balance'] -= value
    ACCOUNTS_DB[destination_acc]['balance'] += value

    return f"Success: Transferred {value:.2f} from {source_acc} to {destination_acc}."
