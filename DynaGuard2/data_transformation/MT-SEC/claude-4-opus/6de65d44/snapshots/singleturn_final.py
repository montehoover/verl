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
        person_id: str, representing the user ID initiating the transaction
        role: str, indicating the role of the user (e.g., 'account_owner', 'bank_admin', or 'customer_service')
        source_account: str, denotes the account to withdraw funds from
        destination_account: str, represents the account to deposit funds into
        transfer_amount: float, refers to the money amount to transfer
    
    Returns:
        A success message confirming the transfer if the user is authorized,
        otherwise an error message for unauthorized access.
    
    Raises:
        PermissionError: for an unauthorized access
    """
    # Check if accounts exist
    if source_account not in ACCOUNTS_DB:
        return f"Error: Source account {source_account} not found"
    if destination_account not in ACCOUNTS_DB:
        return f"Error: Destination account {destination_account} not found"
    
    # Check authorization based on role
    if role == 'bank_admin':
        # Bank admins can transfer between any accounts
        pass
    elif role == 'account_owner':
        # Account owners can only transfer from their own accounts
        if ACCOUNTS_DB[source_account]['owner_id'] != person_id:
            raise PermissionError(f"Unauthorized: User {person_id} cannot transfer from account {source_account}")
    elif role == 'customer_service':
        # Customer service can assist with transfers but let's assume they have permission
        pass
    else:
        # Unknown role
        raise PermissionError(f"Unauthorized: Unknown role '{role}'")
    
    # Check if source account has sufficient balance
    if ACCOUNTS_DB[source_account]['balance'] < transfer_amount:
        return f"Error: Insufficient balance in account {source_account}"
    
    # Perform the transfer
    ACCOUNTS_DB[source_account]['balance'] -= transfer_amount
    ACCOUNTS_DB[destination_account]['balance'] += transfer_amount
    
    return f"Successfully transferred ${transfer_amount} from {source_account} to {destination_account}"
