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

def move_funds(u_id: str, u_role: str, source_account: str, destination_account: str, money: float):
    # Check if both accounts exist
    if source_account not in ACCOUNTS_DB:
        return f"Error: Source account {source_account} not found"
    if destination_account not in ACCOUNTS_DB:
        return f"Error: Destination account {destination_account} not found"
    
    # Get account details
    source_acc = ACCOUNTS_DB[source_account]
    dest_acc = ACCOUNTS_DB[destination_account]
    
    # Check authorization based on role
    if u_role == 'bank_admin' or u_role == 'customer_service':
        # Bank admin and customer service can transfer between any accounts
        pass
    elif u_role == 'account_owner':
        # Account owner can only transfer from their own account
        if source_acc['owner_id'] != u_id:
            raise PermissionError(f"User {u_id} is not authorized to transfer from account {source_account}")
    else:
        # Unknown role
        raise PermissionError(f"Role {u_role} is not authorized to perform transfers")
    
    # Check if source account has sufficient balance
    if source_acc['balance'] < money:
        return f"Error: Insufficient balance in source account {source_account}"
    
    # Check if amount is valid
    if money <= 0:
        return "Error: Transfer amount must be positive"
    
    # Perform the transfer
    source_acc['balance'] -= money
    dest_acc['balance'] += money
    
    return f"Successfully transferred ${money:.2f} from {source_account} to {destination_account}"
