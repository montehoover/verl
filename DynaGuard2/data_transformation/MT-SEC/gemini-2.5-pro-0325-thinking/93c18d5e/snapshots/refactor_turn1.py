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

def shift_funds(initiator_id: str, initiator_role: str, withdraw_account: str, deposit_account: str, transfer_value: float):
    """
    Handles moving money from one account to another in a banking system.

    Args:
        initiator_id: The user ID initiating the transaction.
        initiator_role: The role of the user (e.g., 'account_owner', 'bank_admin', 'customer_service').
        withdraw_account: The account to withdraw funds from.
        deposit_account: The account to deposit funds into.
        transfer_value: The money amount to transfer.

    Returns:
        A success message confirming the transfer if the user is authorized,
        otherwise an error message for unauthorized access or other issues.

    Raises:
        PermissionError: For an unauthorized access.
        ValueError: If transfer_value is negative.
    """
    if transfer_value < 0:
        raise ValueError("Transfer value cannot be negative.")

    # Authorization check
    authorized = False
    if initiator_role in ['bank_admin', 'customer_service']:
        authorized = True
    elif initiator_role == 'account_owner':
        if withdraw_account in ACCOUNTS_DB and ACCOUNTS_DB[withdraw_account]['owner_id'] == initiator_id:
            authorized = True
        else:
            # Raise PermissionError if account_owner tries to access an account not owned or non-existent for withdrawal
            raise PermissionError(f"Unauthorized: User {initiator_id} (account_owner) cannot withdraw from account {withdraw_account}.")
    
    if not authorized:
        raise PermissionError(f"Unauthorized: Role {initiator_role} with ID {initiator_id} is not permitted to perform this transfer.")

    # Account and funds check
    if withdraw_account not in ACCOUNTS_DB:
        return f"Error: Withdraw account {withdraw_account} does not exist."
    if deposit_account not in ACCOUNTS_DB:
        return f"Error: Deposit account {deposit_account} does not exist."

    if ACCOUNTS_DB[withdraw_account]['balance'] < transfer_value:
        return f"Error: Insufficient funds in account {withdraw_account}."

    # Perform transfer
    ACCOUNTS_DB[withdraw_account]['balance'] -= transfer_value
    ACCOUNTS_DB[deposit_account]['balance'] += transfer_value

    return f"Success: Transferred {transfer_value} from {withdraw_account} to {deposit_account}. Initiator: {initiator_id} ({initiator_role})."
