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

def execute_transfer(employee_id: str, role_value: str, source_acct: str, target_acct: str, total_amount: float) -> str:
    # Check if source and target accounts exist
    if source_acct not in ACCOUNTS_DB:
        raise PermissionError(f"Source account {source_acct} not found")
    if target_acct not in ACCOUNTS_DB:
        raise PermissionError(f"Target account {target_acct} not found")
    
    # Get account details
    source_account = ACCOUNTS_DB[source_acct]
    target_account = ACCOUNTS_DB[target_acct]
    
    # Authorization logic based on role
    authorized = False
    
    if role_value == 'bank_admin':
        # Bank admins can transfer between any accounts
        authorized = True
    elif role_value == 'customer_service':
        # Customer service can transfer between any accounts
        authorized = True
    elif role_value == 'account_owner':
        # Account owners can only transfer from their own account
        if source_account['owner_id'] == employee_id:
            authorized = True
    
    if not authorized:
        raise PermissionError(f"User {employee_id} with role {role_value} is not authorized to transfer from account {source_acct}")
    
    # Check if source account has sufficient balance
    if source_account['balance'] < total_amount:
        return f"Error: Insufficient balance in account {source_acct}"
    
    # Perform the transfer
    source_account['balance'] -= total_amount
    target_account['balance'] += total_amount
    
    return f"Successfully transferred ${total_amount:.2f} from {source_acct} to {target_acct}"
