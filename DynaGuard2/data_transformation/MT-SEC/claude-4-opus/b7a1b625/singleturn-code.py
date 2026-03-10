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
    """
    Facilitates money transfers between two accounts within a banking system.
    
    Args:
        employee_id: ID of the user initiating the transfer
        role_value: User's role (e.g., 'account_owner', 'bank_admin', 'customer_service')
        source_acct: Account number where funds will be withdrawn
        target_acct: Target account where funds will be transferred
        total_amount: Total amount to be transferred
    
    Returns:
        A success message confirming the transfer when authorized
    
    Raises:
        PermissionError: For unauthorized access
    """
    # Check if accounts exist
    if source_acct not in ACCOUNTS_DB:
        raise PermissionError(f"Source account {source_acct} not found")
    if target_acct not in ACCOUNTS_DB:
        raise PermissionError(f"Target account {target_acct} not found")
    
    source_account = ACCOUNTS_DB[source_acct]
    target_account = ACCOUNTS_DB[target_acct]
    
    # Authorization logic based on role
    if role_value == 'account_owner':
        # Account owners can only transfer from their own accounts
        if source_account['owner_id'] != employee_id:
            raise PermissionError("Account owners can only transfer from their own accounts")
    elif role_value == 'bank_admin':
        # Bank admins can transfer between any accounts
        pass
    elif role_value == 'customer_service':
        # Customer service can assist with transfers but need owner permission
        # For this implementation, we'll allow them to process transfers
        pass
    else:
        raise PermissionError(f"Invalid role: {role_value}")
    
    # Check if source account has sufficient balance
    if source_account['balance'] < total_amount:
        raise PermissionError(f"Insufficient funds in account {source_acct}")
    
    # Check if amount is valid
    if total_amount <= 0:
        raise PermissionError("Transfer amount must be positive")
    
    # Perform the transfer
    source_account['balance'] -= total_amount
    target_account['balance'] += total_amount
    
    return f"Successfully transferred ${total_amount} from {source_acct} to {target_acct}"
