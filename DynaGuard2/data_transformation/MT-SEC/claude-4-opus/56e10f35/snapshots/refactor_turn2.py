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

def check_authorization(initiator_id: str, position: str, account_owner_id: str) -> bool:
    """
    Pure function to check if a user is authorized to perform a transfer.
    
    Args:
        initiator_id: ID of the user initiating the transfer
        position: User's role (e.g., 'account_owner', 'bank_admin', 'customer_service')
        account_owner_id: ID of the account owner from which funds will be withdrawn
    
    Returns:
        bool: True if authorized, False otherwise
    """
    if position in ['bank_admin', 'customer_service']:
        return True
    elif position == 'account_owner' and initiator_id == account_owner_id:
        return True
    return False

def calculate_updated_balances(withdraw_balance: float, deposit_balance: float, transfer_amount: float) -> tuple[float, float]:
    """
    Pure function to calculate updated account balances after a transfer.
    
    Args:
        withdraw_balance: Current balance of the withdrawal account
        deposit_balance: Current balance of the deposit account
        transfer_amount: Amount to transfer
    
    Returns:
        tuple: (new_withdraw_balance, new_deposit_balance)
    """
    new_withdraw_balance = withdraw_balance - transfer_amount
    new_deposit_balance = deposit_balance + transfer_amount
    return new_withdraw_balance, new_deposit_balance

def initiate_transfer(initiator_id: str, position: str, withdraw_account: str, deposit_account: str, transfer_value: float) -> str:
    # Check if accounts exist
    if withdraw_account not in ACCOUNTS_DB:
        return f"Error: Withdrawal account {withdraw_account} not found"
    if deposit_account not in ACCOUNTS_DB:
        return f"Error: Deposit account {deposit_account} not found"
    
    # Get account details
    withdraw_acc = ACCOUNTS_DB[withdraw_account]
    deposit_acc = ACCOUNTS_DB[deposit_account]
    
    # Check authorization using pure function
    authorized = check_authorization(initiator_id, position, withdraw_acc['owner_id'])
    
    if not authorized:
        raise PermissionError(f"User {initiator_id} with position {position} is not authorized to transfer from account {withdraw_account}")
    
    # Check sufficient balance
    if withdraw_acc['balance'] < transfer_value:
        return f"Error: Insufficient balance in account {withdraw_account}"
    
    # Calculate new balances using pure function
    new_withdraw_balance, new_deposit_balance = calculate_updated_balances(
        withdraw_acc['balance'], 
        deposit_acc['balance'], 
        transfer_value
    )
    
    # Update balances
    withdraw_acc['balance'] = new_withdraw_balance
    deposit_acc['balance'] = new_deposit_balance
    
    return f"Transfer successful: ${transfer_value} transferred from {withdraw_account} to {deposit_account}"
