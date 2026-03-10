# Assuming ACCOUNTS_DB is available as a dictionary with account details
# Example: ACCOUNTS_DB = {'12345': {'balance': 1000.0}, '67890': {'balance': 500.0}}

# Define ACCOUNTS_DB as a module-level variable
ACCOUNTS_DB = {}

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
