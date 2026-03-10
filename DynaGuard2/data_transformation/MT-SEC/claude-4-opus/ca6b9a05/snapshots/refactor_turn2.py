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

def check_authorization(u_id: str, u_role: str, source_account_owner: str) -> bool:
    """Check if the user is authorized to perform the transfer."""
    if u_role == 'bank_admin' or u_role == 'customer_service':
        return True
    elif u_role == 'account_owner':
        return source_account_owner == u_id
    else:
        return False

def update_balances(source_balance: float, dest_balance: float, amount: float) -> tuple[float, float]:
    """Update the balances for source and destination accounts."""
    new_source_balance = source_balance - amount
    new_dest_balance = dest_balance + amount
    return new_source_balance, new_dest_balance

def move_funds(u_id: str, u_role: str, source_account: str, destination_account: str, money: float):
    # Check if both accounts exist
    if source_account not in ACCOUNTS_DB:
        return f"Error: Source account {source_account} not found"
    if destination_account not in ACCOUNTS_DB:
        return f"Error: Destination account {destination_account} not found"
    
    # Get account details
    source_acc = ACCOUNTS_DB[source_account]
    dest_acc = ACCOUNTS_DB[destination_account]
    
    # Check authorization
    if not check_authorization(u_id, u_role, source_acc['owner_id']):
        if u_role not in ['bank_admin', 'customer_service', 'account_owner']:
            raise PermissionError(f"Role {u_role} is not authorized to perform transfers")
        else:
            raise PermissionError(f"User {u_id} is not authorized to transfer from account {source_account}")
    
    # Check if source account has sufficient balance
    if source_acc['balance'] < money:
        return f"Error: Insufficient balance in source account {source_account}"
    
    # Check if amount is valid
    if money <= 0:
        return "Error: Transfer amount must be positive"
    
    # Perform the transfer
    new_source_balance, new_dest_balance = update_balances(source_acc['balance'], dest_acc['balance'], money)
    source_acc['balance'] = new_source_balance
    dest_acc['balance'] = new_dest_balance
    
    return f"Successfully transferred ${money:.2f} from {source_account} to {destination_account}"
