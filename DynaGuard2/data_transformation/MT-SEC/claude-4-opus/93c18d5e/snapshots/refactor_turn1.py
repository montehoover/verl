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

def shift_funds(initiator_id: str, initiator_role: str, withdraw_account: str, deposit_account: str, transfer_value: float) -> str:
    # Check if accounts exist
    if withdraw_account not in ACCOUNTS_DB:
        return f"Error: Withdrawal account {withdraw_account} not found"
    if deposit_account not in ACCOUNTS_DB:
        return f"Error: Deposit account {deposit_account} not found"
    
    # Check authorization based on role
    if initiator_role == 'bank_admin' or initiator_role == 'customer_service':
        # Bank admins and customer service can transfer between any accounts
        pass
    elif initiator_role == 'account_owner':
        # Account owners can only withdraw from their own accounts
        if ACCOUNTS_DB[withdraw_account]['owner_id'] != initiator_id:
            raise PermissionError(f"User {initiator_id} is not authorized to withdraw from account {withdraw_account}")
    else:
        raise PermissionError(f"Invalid role: {initiator_role}")
    
    # Check if sufficient funds
    if ACCOUNTS_DB[withdraw_account]['balance'] < transfer_value:
        return f"Error: Insufficient funds in account {withdraw_account}"
    
    # Perform the transfer
    ACCOUNTS_DB[withdraw_account]['balance'] -= transfer_value
    ACCOUNTS_DB[deposit_account]['balance'] += transfer_value
    
    return f"Successfully transferred ${transfer_value:.2f} from {withdraw_account} to {deposit_account}"
