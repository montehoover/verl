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

def execute_transfer(employee_id: str, role_value: str, source_acct: str, target_acct: str, total_amount: float):
    """
    Facilitates money transfers between two accounts within a banking system.

    Args:
        employee_id: ID of the user initiating the transfer.
        role_value: User's role (e.g., 'account_owner', 'bank_admin', 'customer_service').
        source_acct: Account number from which funds will be withdrawn.
        target_acct: Account number to which funds will be transferred.
        total_amount: The total amount to be transferred.

    Returns:
        A success message confirming the transfer when authorized.

    Raises:
        PermissionError: For unauthorized access or if preconditions for transfer are not met.
    """

    # Role-based authorization and initial checks
    if role_value == 'account_owner':
        if source_acct not in ACCOUNTS_DB:
            raise PermissionError(f"Authorization failed: Source account {source_acct} does not exist.")
        if ACCOUNTS_DB[source_acct]['owner_id'] != employee_id:
            raise PermissionError("Authorization failed: User is not the owner of the source account.")
    elif role_value in ['bank_admin', 'customer_service']:
        # Authorized by role, but accounts must exist.
        # Check source account existence for these roles.
        if source_acct not in ACCOUNTS_DB:
            raise PermissionError(f"Authorization failed: Source account {source_acct} does not exist.")
    else: # Invalid or unauthorized role
        raise PermissionError(f"Authorization failed: Invalid role '{role_value}'.")

    # General pre-transfer condition checks
    # Source account existence is confirmed by this point for all authorized roles.

    if target_acct not in ACCOUNTS_DB:
        raise PermissionError(f"Authorization failed: Target account {target_acct} does not exist.")

    if not isinstance(total_amount, (int, float)) or total_amount <= 0:
        raise PermissionError("Authorization failed: Transfer amount must be a positive number.")

    source_account_details = ACCOUNTS_DB[source_acct]

    if source_account_details['balance'] < total_amount:
        raise PermissionError(f"Authorization failed: Insufficient funds in account {source_acct}.")

    # Perform the transfer
    # Ensure we are modifying the actual mutable dictionary entries within ACCOUNTS_DB
    ACCOUNTS_DB[source_acct]['balance'] -= total_amount
    ACCOUNTS_DB[target_acct]['balance'] += total_amount

    return f"Successfully transferred ${total_amount:.1f} from {source_acct} to {target_acct}"
