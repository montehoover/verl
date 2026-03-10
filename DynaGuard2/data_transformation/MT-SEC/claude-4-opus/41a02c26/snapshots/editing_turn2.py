# Assume ACCOUNTS_DB is available with account details
# Example: ACCOUNTS_DB = {'123456': {'balance': 1000.00}, '789012': {'balance': 500.00}}

# Mock ACCOUNTS_DB for testing/demonstration
ACCOUNTS_DB = {}

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
        if user_id in USER_PERMISSIONS_DB:
            user_data = USER_PERMISSIONS_DB[user_id]
            if 'accounts' in user_data and account_number in user_data['accounts']:
                return True
    
    return False
