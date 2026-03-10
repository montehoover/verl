# Assume ACCOUNTS_DB is available with account details
# Example: ACCOUNTS_DB = {'123456': {'balance': 1000.00}, '789012': {'balance': 500.00}}

# Mock ACCOUNTS_DB for testing/demonstration
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

# Mock USER_PERMISSIONS_DB for testing/demonstration
# Example: USER_PERMISSIONS_DB = {'user123': {'role': 'account_owner', 'accounts': ['123456']}}
USER_PERMISSIONS_DB = {}

def has_sufficient_funds(account_number, amount):
    """
    Check if an account has sufficient funds for a transaction.
    
    Args:
        account_number (str): The account number to check
        amount (float): The amount to check against the balance
        
    Returns:
        bool: True if account has sufficient funds, False otherwise
    """
    if account_number in ACCOUNTS_DB:
        account_balance = ACCOUNTS_DB[account_number]['balance']
        return account_balance >= amount
    return False

def verify_user_role(user_id, role, account_number):
    """
    Check if a user has the right permissions to access an account.
    
    Args:
        user_id (str): The user ID to check
        role (str): The role of the user
        account_number (str): The account number to access
        
    Returns:
        bool: True if user has necessary permissions, False otherwise
    """
    # Bank admins can access any account
    if role == 'bank_admin':
        return True
    
    # Customer service can access any account
    if role == 'customer_service':
        return True
    
    # Account owners can only access their own accounts
    if role == 'account_owner':
        if account_number in ACCOUNTS_DB:
            account_data = ACCOUNTS_DB[account_number]
            if 'owner_id' in account_data and account_data['owner_id'] == user_id:
                return True
    
    return False

def process_transfer(uid, role, source_account, destination_account, transfer_amount):
    """
    Handle money transfers between accounts.
    
    Args:
        uid (str): The user ID initiating the transfer
        role (str): The user's role
        source_account (str): The account to withdraw from
        destination_account (str): The account to deposit into
        transfer_amount (float): The amount to transfer
        
    Returns:
        str: Success message if authorized, error message if failed
    """
    # Verify user has permission to access source account
    if not verify_user_role(uid, role, source_account):
        return "Error: Unauthorized access to source account"
    
    # Check if source account exists
    if source_account not in ACCOUNTS_DB:
        return "Error: Source account not found"
    
    # Check if destination account exists
    if destination_account not in ACCOUNTS_DB:
        return "Error: Destination account not found"
    
    # Check if source account has sufficient funds
    if not has_sufficient_funds(source_account, transfer_amount):
        return "Error: Insufficient funds in source account"
    
    # Perform the transfer
    ACCOUNTS_DB[source_account]['balance'] -= transfer_amount
    ACCOUNTS_DB[destination_account]['balance'] += transfer_amount
    
    return f"Success: Transferred ${transfer_amount:.2f} from {source_account} to {destination_account}"
