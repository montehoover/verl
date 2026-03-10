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

def check_authorization(identifier: str, role: str, source_owner_id: str) -> bool:
    """Check if the transfer is authorized based on role and ownership."""
    if role in ['bank_admin', 'customer_service']:
        return True
    elif role == 'account_owner':
        return source_owner_id == identifier
    else:
        return False

def update_balances(source_account: dict, destination_account: dict, value: float) -> None:
    """Update the balances of source and destination accounts."""
    source_account['balance'] -= value
    destination_account['balance'] += value

def execute_transfer(identifier: str, role: str, source_acc: str, destination_acc: str, value: float) -> str:
    # Check if source and destination accounts exist
    if source_acc not in ACCOUNTS_DB:
        return f"Error: Source account {source_acc} not found"
    if destination_acc not in ACCOUNTS_DB:
        return f"Error: Destination account {destination_acc} not found"
    
    source_account = ACCOUNTS_DB[source_acc]
    destination_account = ACCOUNTS_DB[destination_acc]
    
    # Authorization check
    if not check_authorization(identifier, role, source_account['owner_id']):
        if role not in ['bank_admin', 'customer_service', 'account_owner']:
            raise PermissionError(f"Unauthorized: Unknown role {role}")
        else:
            raise PermissionError(f"Unauthorized: {identifier} cannot transfer from account {source_acc}")
    
    # Check if source account has sufficient balance
    if source_account['balance'] < value:
        return f"Error: Insufficient funds in account {source_acc}"
    
    # Perform the transfer
    update_balances(source_account, destination_account, value)
    
    return f"Success: Transferred {value} from {source_acc} to {destination_acc}"
