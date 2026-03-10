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

def send_funds(account_holder: str, user_role: str, withdrawal_account: str, receiving_account: str, fund_amount: float) -> str:
    # Check if both accounts exist
    if withdrawal_account not in ACCOUNTS_DB:
        return f"Error: Withdrawal account {withdrawal_account} not found"
    if receiving_account not in ACCOUNTS_DB:
        return f"Error: Receiving account {receiving_account} not found"
    
    # Check authorization based on user role
    if user_role == 'bank_admin' or user_role == 'customer_service':
        # Bank admins and customer service can transfer between any accounts
        pass
    elif user_role == 'account_owner':
        # Account owners can only withdraw from their own accounts
        if ACCOUNTS_DB[withdrawal_account]['owner_id'] != account_holder:
            raise PermissionError(f"Unauthorized access: User {account_holder} cannot withdraw from account {withdrawal_account}")
    else:
        # Unknown role
        raise PermissionError(f"Unauthorized access: Unknown role {user_role}")
    
    # Check if withdrawal account has sufficient funds
    if ACCOUNTS_DB[withdrawal_account]['balance'] < fund_amount:
        return f"Error: Insufficient funds in account {withdrawal_account}"
    
    # Perform the transfer
    ACCOUNTS_DB[withdrawal_account]['balance'] -= fund_amount
    ACCOUNTS_DB[receiving_account]['balance'] += fund_amount
    
    return f"Success: Transferred ${fund_amount:.2f} from {withdrawal_account} to {receiving_account}"
