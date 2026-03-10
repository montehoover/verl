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

def initiate_transfer(initiator_id: str, position: str, withdraw_account: str, deposit_account: str, transfer_value: float) -> str:
    # Check if accounts exist
    if withdraw_account not in ACCOUNTS_DB:
        return f"Error: Withdrawal account {withdraw_account} not found"
    if deposit_account not in ACCOUNTS_DB:
        return f"Error: Deposit account {deposit_account} not found"
    
    # Get account details
    withdraw_acc = ACCOUNTS_DB[withdraw_account]
    deposit_acc = ACCOUNTS_DB[deposit_account]
    
    # Authorization logic
    authorized = False
    
    # Bank admins can transfer between any accounts
    if position == 'bank_admin':
        authorized = True
    # Customer service can transfer between any accounts
    elif position == 'customer_service':
        authorized = True
    # Account owners can only transfer from their own accounts
    elif position == 'account_owner':
        if withdraw_acc['owner_id'] == initiator_id:
            authorized = True
    
    if not authorized:
        raise PermissionError(f"User {initiator_id} with position {position} is not authorized to transfer from account {withdraw_account}")
    
    # Check sufficient balance
    if withdraw_acc['balance'] < transfer_value:
        return f"Error: Insufficient balance in account {withdraw_account}"
    
    # Perform transfer
    withdraw_acc['balance'] -= transfer_value
    deposit_acc['balance'] += transfer_value
    
    return f"Transfer successful: ${transfer_value} transferred from {withdraw_account} to {deposit_account}"
