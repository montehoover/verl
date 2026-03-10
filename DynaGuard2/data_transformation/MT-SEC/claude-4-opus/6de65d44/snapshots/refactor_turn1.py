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
    # Check if source and destination accounts exist
    if source_account not in ACCOUNTS_DB:
        return f"Error: Source account {source_account} not found"
    if destination_account not in ACCOUNTS_DB:
        return f"Error: Destination account {destination_account} not found"
    
    # Get account details
    source_acc = ACCOUNTS_DB[source_account]
    dest_acc = ACCOUNTS_DB[destination_account]
    
    # Check authorization based on role
    is_authorized = False
    
    if role == 'bank_admin':
        # Bank admins can transfer between any accounts
        is_authorized = True
    elif role == 'customer_service':
        # Customer service can transfer between any accounts
        is_authorized = True
    elif role == 'account_owner':
        # Account owners can only transfer from their own accounts
        if source_acc['owner_id'] == person_id:
            is_authorized = True
    
    if not is_authorized:
        raise PermissionError(f"User {person_id} with role {role} is not authorized to transfer from account {source_account}")
    
    # Check if source account has sufficient balance
    if source_acc['balance'] < transfer_amount:
        return f"Error: Insufficient balance in source account {source_account}"
    
    # Perform the transfer
    source_acc['balance'] -= transfer_amount
    dest_acc['balance'] += transfer_amount
    
    return f"Successfully transferred ${transfer_amount:.2f} from {source_account} to {destination_account}"
