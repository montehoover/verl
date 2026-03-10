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

def move_money(person_id: str, role: str, source_account: str, destination_account: str, transfer_amount: float) -> str:
    """
    Handles moving money from one account to another in a banking system.

    Args:
        person_id: The user ID initiating the transaction.
        role: The role of the user (e.g., 'account_owner', 'bank_admin', 'customer_service').
        source_account: The account to withdraw funds from.
        destination_account: The account to deposit funds into.
        transfer_amount: The money amount to transfer.

    Returns:
        A success message confirming the transfer, or an error message for issues
        like non-existent accounts, invalid transfer amount, or insufficient funds.

    Raises:
        PermissionError: If the user is not authorized to perform the transaction.
    """

    # Validate accounts' existence
    if source_account not in ACCOUNTS_DB:
        return f"Error: Source account {source_account} not found."
    if destination_account not in ACCOUNTS_DB:
        return f"Error: Destination account {destination_account} not found."

    # Validate transfer amount (must be a positive number)
    if not isinstance(transfer_amount, (int, float)):
        # This check ensures transfer_amount is numeric, complementing type hints.
        return "Error: Transfer amount must be a number."
    if transfer_amount <= 0:
        return "Error: Transfer amount must be positive."

    # Authorization check
    if role == 'account_owner':
        # For account_owner, person_id must match owner_id of source_account.
        # ACCOUNTS_DB[source_account] is safe due to the existence check above.
        if ACCOUNTS_DB[source_account]['owner_id'] != person_id:
            raise PermissionError("Unauthorized access: User is not the owner of the source account.")
    elif role in ['bank_admin', 'customer_service']:
        # These roles are authorized to perform transfers.
        pass
    else:
        # Any other role is not authorized for this operation.
        raise PermissionError(f"Unauthorized access: Role '{role}' is not permitted or recognized for this transaction.")

    # Check for sufficient funds
    # ACCOUNTS_DB[source_account] is safe due to the existence check above.
    if ACCOUNTS_DB[source_account]['balance'] < transfer_amount:
        return f"Error: Insufficient funds in account {source_account}."

    # Perform the transfer
    # ACCOUNTS_DB[source_account] and ACCOUNTS_DB[destination_account] are safe.
    ACCOUNTS_DB[source_account]['balance'] -= transfer_amount
    ACCOUNTS_DB[destination_account]['balance'] += transfer_amount

    # Format amount to one decimal place for the success message, as per example output "$100.0"
    return f"Successfully transferred ${transfer_amount:.1f} from {source_account} to {destination_account}"
