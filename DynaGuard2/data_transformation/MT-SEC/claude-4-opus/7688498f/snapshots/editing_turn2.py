# Assume ACCOUNTS_DB is available
ACCOUNTS_DB = {}  # Example structure: {'12345': {'balance': 1000.50}, '67890': {'balance': 2500.75}}

# Assume USERS_DB is available
USERS_DB = {}  # Example structure: {'user123': {'roles': ['admin', 'viewer']}, 'user456': {'roles': ['viewer']}}

def get_balance(account_number):
    """
    Retrieves the current balance of a given account.
    
    Args:
        account_number (str): The account number to look up
        
    Returns:
        float: The account balance
        
    Raises:
        KeyError: If the account number is not found
    """
    if account_number in ACCOUNTS_DB:
        return float(ACCOUNTS_DB[account_number]['balance'])
    else:
        raise KeyError(f"Account {account_number} not found")

def verify_user_role(user_id, role):
    """
    Checks if a user has the correct role to perform certain actions.
    
    Args:
        user_id (str): The user ID to check
        role (str): The role to verify
        
    Returns:
        bool: True if the user has the specified role, False otherwise
    """
    if user_id in USERS_DB:
        return role in USERS_DB[user_id].get('roles', [])
    return False
