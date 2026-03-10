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

def check_authorization(u_identifier: str, u_role: str, from_account: dict) -> bool:
    """Check if the user is authorized to perform the transfer.
    
    Returns True if authorized, raises PermissionError if not.
    """
    if u_role == 'account_owner':
        if from_account['owner_id'] != u_identifier:
            raise PermissionError(f"User {u_identifier} is not authorized to transfer from account {from_account['account_number']}")
        return True
    elif u_role in ['bank_admin', 'customer_service']:
        return True
    else:
        raise PermissionError(f"Role {u_role} is not authorized to perform transfers")

def update_balances(from_account: dict, to_account: dict, amount: float) -> tuple[float, float]:
    """Update account balances and return the new balances.
    
    Returns a tuple of (new_from_balance, new_to_balance).
    """
    new_from_balance = from_account['balance'] - amount
    new_to_balance = to_account['balance'] + amount
    return new_from_balance, new_to_balance

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
    check_authorization(u_identifier, u_role, from_account)
    
    # Check sufficient balance
    if from_account['balance'] < transfer_amount:
        return f"Error: Insufficient balance in account {account_from}"
    
    # Check for valid transfer amount
    if transfer_amount <= 0:
        return "Error: Transfer amount must be positive"
    
    # Perform the transfer
    new_from_balance, new_to_balance = update_balances(from_account, to_account, transfer_amount)
    from_account['balance'] = new_from_balance
    to_account['balance'] = new_to_balance
    
    return f"Success: Transferred ${transfer_amount:.2f} from {account_from} to {account_to}"
