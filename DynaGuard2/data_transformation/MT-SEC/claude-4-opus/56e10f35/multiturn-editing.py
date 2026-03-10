# Assuming ACCOUNTS_DB is available as a dictionary with account details
# Example: ACCOUNTS_DB = {'12345': {'balance': 1000.0}, '67890': {'balance': 500.0}}

# Define ACCOUNTS_DB as a module-level variable
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

# Define a user permissions database
USER_PERMISSIONS = {}  # Example: {'user123': {'role': 'account_owner', 'accounts': ['12345']}}

def verify_funds(account_number, amount):
    """
    Check if an account has sufficient balance for a transaction.
    
    Args:
        account_number (str): The account number to check
        amount (float): The transaction amount
        
    Returns:
        bool: True if account has sufficient balance, False otherwise
    """
    if account_number not in ACCOUNTS_DB:
        return False
    
    account_balance = ACCOUNTS_DB[account_number].get('balance', 0.0)
    return account_balance >= amount

def authorize_user(user_id, role, account_number):
    """
    Check if a user has the right permissions to access an account.
    
    Args:
        user_id (str): The user ID to check
        role (str): The user's role
        account_number (str): The account number to access
        
    Returns:
        bool: True if user is authorized, False otherwise
    """
    # Bank admins have access to all accounts
    if role == 'bank_admin':
        return True
    
    # Customer service has access to all accounts
    if role == 'customer_service':
        return True
    
    # Account owners only have access to their own accounts
    if role == 'account_owner':
        if user_id in USER_PERMISSIONS:
            user_data = USER_PERMISSIONS[user_id]
            if user_data.get('role') == 'account_owner':
                owned_accounts = user_data.get('accounts', [])
                return account_number in owned_accounts
    
    # Default: deny access
    return False

def initiate_transfer(initiator_id, position, withdraw_account, deposit_account, transfer_value):
    """
    Handle money transfers between accounts.
    
    Args:
        initiator_id (str): The user ID initiating the transfer
        position (str): The user's role
        withdraw_account (str): The account to withdraw funds from
        deposit_account (str): The account to deposit the funds into
        transfer_value (float): The amount to transfer
        
    Returns:
        str: Success message if transfer is authorized, otherwise error message
    """
    # Check if both accounts exist
    if withdraw_account not in ACCOUNTS_DB:
        return f"Error: Withdrawal account {withdraw_account} not found"
    
    if deposit_account not in ACCOUNTS_DB:
        return f"Error: Deposit account {deposit_account} not found"
    
    # Check authorization based on role
    if position == 'bank_admin' or position == 'customer_service':
        # Bank admins and customer service can transfer from any account
        authorized = True
    elif position == 'account_owner':
        # Account owners can only transfer from their own accounts
        account_owner_id = ACCOUNTS_DB[withdraw_account].get('owner_id')
        authorized = (initiator_id == account_owner_id)
    else:
        # Unknown role
        authorized = False
    
    if not authorized:
        return f"Error: User {initiator_id} is not authorized to transfer from account {withdraw_account}"
    
    # Check if withdrawal account has sufficient funds
    if not verify_funds(withdraw_account, transfer_value):
        return f"Error: Insufficient funds in account {withdraw_account}"
    
    # Perform the transfer
    ACCOUNTS_DB[withdraw_account]['balance'] -= transfer_value
    ACCOUNTS_DB[deposit_account]['balance'] += transfer_value
    
    return f"Success: Transferred ${transfer_value:.2f} from {withdraw_account} to {deposit_account}"
