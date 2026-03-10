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

def send_funds(account_holder, user_role, withdrawal_account, receiving_account, fund_amount):
    """
    Handles moving money from one account to another in a banking system.
    
    Args:
        account_holder: str, representing the user ID initiating the transaction
        user_role: str, indicating the role of the user (e.g., 'account_owner', 'bank_admin', or 'customer_service')
        withdrawal_account: str, denotes the account to withdraw funds from
        receiving_account: str, represents the account to deposit funds into
        fund_amount: float, refers to the money amount to transfer
    
    Returns:
        A success message confirming the transfer if the user is authorized,
        otherwise an error message for unauthorized access.
    
    Raises:
        PermissionError: for an unauthorized access
    """
    # Check if both accounts exist
    if withdrawal_account not in ACCOUNTS_DB:
        return f"Error: Withdrawal account {withdrawal_account} not found"
    if receiving_account not in ACCOUNTS_DB:
        return f"Error: Receiving account {receiving_account} not found"
    
    # Get account information
    withdraw_acc_info = ACCOUNTS_DB[withdrawal_account]
    receive_acc_info = ACCOUNTS_DB[receiving_account]
    
    # Check authorization based on user role
    is_authorized = False
    
    if user_role == 'bank_admin':
        # Bank admins can transfer between any accounts
        is_authorized = True
    elif user_role == 'customer_service':
        # Customer service can transfer between any accounts
        is_authorized = True
    elif user_role == 'account_owner':
        # Account owners can only transfer from their own accounts
        if withdraw_acc_info['owner_id'] == account_holder:
            is_authorized = True
    
    # If not authorized, raise PermissionError
    if not is_authorized:
        raise PermissionError(f"User {account_holder} with role {user_role} is not authorized to transfer from account {withdrawal_account}")
    
    # Check if withdrawal account has sufficient funds
    if withdraw_acc_info['balance'] < fund_amount:
        return f"Error: Insufficient funds in account {withdrawal_account}"
    
    # Perform the transfer
    ACCOUNTS_DB[withdrawal_account]['balance'] -= fund_amount
    ACCOUNTS_DB[receiving_account]['balance'] += fund_amount
    
    return f"Successfully transferred ${fund_amount} from {withdrawal_account} to {receiving_account}"
