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

def fund_transfer(u_identifier: str, u_role: str, account_from: str, account_to: str, transfer_amount: float):
    # Check if accounts exist
    if account_from not in ACCOUNTS_DB:
        return f"Error: Source account {account_from} not found"
    if account_to not in ACCOUNTS_DB:
        return f"Error: Destination account {account_to} not found"
    
    # Get account details
    from_account = ACCOUNTS_DB[account_from]
    to_account = ACCOUNTS_DB[account_to]
    
    # Authorization checks
    if u_role == 'account_owner':
        # Account owner can only transfer from their own account
        if from_account['owner_id'] != u_identifier:
            raise PermissionError(f"User {u_identifier} is not authorized to transfer from account {account_from}")
    elif u_role in ['bank_admin', 'customer_service']:
        # Bank admin and customer service can transfer from any account
        pass
    else:
        # Unknown role
        raise PermissionError(f"Role {u_role} is not authorized to perform transfers")
    
    # Check sufficient balance
    if from_account['balance'] < transfer_amount:
        return f"Error: Insufficient balance in account {account_from}"
    
    # Check for valid transfer amount
    if transfer_amount <= 0:
        return "Error: Transfer amount must be positive"
    
    # Perform the transfer
    from_account['balance'] -= transfer_amount
    to_account['balance'] += transfer_amount
    
    return f"Success: Transferred ${transfer_amount:.2f} from {account_from} to {account_to}"
