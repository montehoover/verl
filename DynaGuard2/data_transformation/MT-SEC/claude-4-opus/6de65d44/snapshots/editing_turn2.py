# Assume ACCOUNTS_DB is defined elsewhere in the application
# For example: ACCOUNTS_DB = {'12345': {'balance': 1000.0}, '67890': {'balance': 500.0}}

# Mock database for accounts (this would normally be imported from another module)
ACCOUNTS_DB = {}

# Mock database for user permissions (this would normally be imported from another module)
USER_PERMISSIONS = {}

def has_sufficient_funds(account_number, amount):
    """
    Check if an account has sufficient funds for a transaction.
    
    Args:
        account_number (str): The account number to check
        amount (float): The amount to check against the account balance
        
    Returns:
        bool: True if account balance >= amount, False otherwise
    """
    if account_number not in ACCOUNTS_DB:
        return False
    
    account_balance = ACCOUNTS_DB[account_number].get('balance', 0)
    return account_balance >= amount


def verify_user_role(user_id, role, account_number):
    """
    Verify if a user has the right permissions to access or modify an account.
    
    Args:
        user_id (str): The user ID to verify
        role (str): The user's role (e.g., 'account_owner', 'bank_admin', 'customer_service')
        account_number (str): The account number to access
        
    Returns:
        bool: True if user has necessary permissions, False otherwise
    """
    # Bank admins can access any account
    if role == 'bank_admin':
        return True
    
    # Customer service can access any account (read-only typically, but permissions check here)
    if role == 'customer_service':
        return True
    
    # Account owners can only access their own accounts
    if role == 'account_owner':
        if account_number not in ACCOUNTS_DB:
            return False
        
        # Check if the user_id matches the account owner
        account_owner_id = ACCOUNTS_DB[account_number].get('owner_id')
        return user_id == account_owner_id
    
    # Any other role or undefined role has no access
    return False
