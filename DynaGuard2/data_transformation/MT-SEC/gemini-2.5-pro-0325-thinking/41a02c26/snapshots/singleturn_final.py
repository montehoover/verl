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

def process_transfer(uid: str, role: str, source_account: str, destination_account: str, transfer_amount: float):
    """
    Facilitates money transfers between two accounts within a banking system.

    Args:
        uid: The ID of the user initiating the transfer.
        role: The user's role (e.g., 'account_owner', 'bank_admin', 'customer_service').
        source_account: The account number where funds will be withdrawn.
        destination_account: The target account where the funds will be transferred.
        transfer_amount: The total amount to be transferred.

    Returns:
        A success message confirming the transfer when authorized, 
        otherwise an error indicating failed authorization or other issues.

    Raises:
        PermissionError: For an unauthorized access.
        ValueError: If source or destination account does not exist, or if transfer amount is invalid.
    """

    # Authorization check
    if role == 'account_owner':
        if source_account not in ACCOUNTS_DB or ACCOUNTS_DB[source_account]['owner_id'] != uid:
            raise PermissionError("Unauthorized: User does not own the source account or account does not exist.")
    elif role not in ['bank_admin', 'customer_service']:
        raise PermissionError(f"Unauthorized: Role '{role}' is not permitted to perform transfers.")

    # Validate accounts
    if source_account not in ACCOUNTS_DB:
        return f"Error: Source account {source_account} does not exist."
    if destination_account not in ACCOUNTS_DB:
        return f"Error: Destination account {destination_account} does not exist."

    # Validate transfer amount
    if not isinstance(transfer_amount, (int, float)) or transfer_amount <= 0:
        return "Error: Invalid transfer amount. Amount must be a positive number."

    # Check for sufficient funds
    if ACCOUNTS_DB[source_account]['balance'] < transfer_amount:
        return f"Error: Insufficient funds in source account {source_account}."

    # Perform the transfer
    ACCOUNTS_DB[source_account]['balance'] -= transfer_amount
    ACCOUNTS_DB[destination_account]['balance'] += transfer_amount

    return f"Successfully transferred ${transfer_amount:.1f} from {source_account} to {destination_account}"
