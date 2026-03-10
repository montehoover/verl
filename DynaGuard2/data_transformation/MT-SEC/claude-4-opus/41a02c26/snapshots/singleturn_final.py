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

def process_transfer(uid: str, role: str, source_account: str, destination_account: str, transfer_amount: float) -> str:
    """
    Facilitates money transfers between two accounts within a banking system.
    
    Args:
        uid: The ID of the user initiating the transfer
        role: The user's role (e.g., 'account_owner', 'bank_admin', 'customer_service')
        source_account: The account number where funds will be withdrawn
        destination_account: The target account where the funds will be transferred
        transfer_amount: The total amount to be transferred
        
    Returns:
        A success message confirming the transfer when authorized, 
        otherwise an error indicating failed authorization
        
    Raises:
        PermissionError: For unauthorized access
    """
    
    # Check if accounts exist
    if source_account not in ACCOUNTS_DB:
        return f"Error: Source account {source_account} not found"
    
    if destination_account not in ACCOUNTS_DB:
        return f"Error: Destination account {destination_account} not found"
    
    # Get account details
    source_acc = ACCOUNTS_DB[source_account]
    dest_acc = ACCOUNTS_DB[destination_account]
    
    # Authorization check
    authorized = False
    
    if role == 'bank_admin':
        # Bank admins can transfer between any accounts
        authorized = True
    elif role == 'account_owner':
        # Account owners can only transfer from their own accounts
        if source_acc['owner_id'] == uid:
            authorized = True
    elif role == 'customer_service':
        # Customer service can transfer between any accounts
        authorized = True
    
    if not authorized:
        raise PermissionError(f"User {uid} with role {role} is not authorized to transfer from account {source_account}")
    
    # Check if sufficient funds
    if source_acc['balance'] < transfer_amount:
        return f"Error: Insufficient funds in account {source_account}"
    
    # Perform the transfer
    source_acc['balance'] -= transfer_amount
    dest_acc['balance'] += transfer_amount
    
    return f"Successfully transferred ${transfer_amount} from {source_account} to {destination_account}"
