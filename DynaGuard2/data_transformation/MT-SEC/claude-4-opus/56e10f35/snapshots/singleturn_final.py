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

def initiate_transfer(initiator_id: str, position: str, withdraw_account: str, deposit_account: str, transfer_value: float) -> str:
    """
    Facilitates money transfers between two accounts within a banking system.
    
    Args:
        initiator_id: ID of the user initiating the transfer
        position: User's role (e.g., 'account_owner', 'bank_admin', 'customer_service')
        withdraw_account: Account number where funds will be withdrawn
        deposit_account: Target account where funds will be transferred
        transfer_value: Total amount to be transferred
    
    Returns:
        A success message confirming the transfer when authorized
    
    Raises:
        PermissionError: For unauthorized access
    """
    # Check if accounts exist
    if withdraw_account not in ACCOUNTS_DB:
        raise PermissionError(f"Account {withdraw_account} does not exist")
    if deposit_account not in ACCOUNTS_DB:
        raise PermissionError(f"Account {deposit_account} does not exist")
    
    # Get account details
    withdraw_acc = ACCOUNTS_DB[withdraw_account]
    deposit_acc = ACCOUNTS_DB[deposit_account]
    
    # Authorization checks based on position
    if position == 'account_owner':
        # Account owner can only transfer from their own account
        if withdraw_acc['owner_id'] != initiator_id:
            raise PermissionError("Account owners can only transfer from their own accounts")
    elif position in ['bank_admin', 'customer_service']:
        # Bank admin and customer service can transfer between any accounts
        pass
    else:
        # Unknown position
        raise PermissionError(f"Unknown position: {position}")
    
    # Check if sufficient balance
    if withdraw_acc['balance'] < transfer_value:
        raise PermissionError(f"Insufficient balance in account {withdraw_account}")
    
    # Perform the transfer
    withdraw_acc['balance'] -= transfer_value
    deposit_acc['balance'] += transfer_value
    
    return f"Successfully transferred ${transfer_value} from {withdraw_account} to {deposit_account}"
