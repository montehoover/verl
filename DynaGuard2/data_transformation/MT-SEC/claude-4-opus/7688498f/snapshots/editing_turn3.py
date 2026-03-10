# Assume ACCOUNTS_DB is available
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

def fund_transfer(u_identifier, u_role, account_from, account_to, transfer_amount):
    """
    Manages money transfers between accounts.
    
    Args:
        u_identifier (str): The user requesting the transfer
        u_role (str): The user's role
        account_from (str): The account to withdraw from
        account_to (str): The account to deposit into
        transfer_amount (float): The amount to transfer
        
    Returns:
        str: Success message if authorized and completed, error message otherwise
    """
    # Check if user has permission (admin role or owns the source account)
    if u_role != 'admin' and (account_from not in ACCOUNTS_DB or ACCOUNTS_DB[account_from]['owner_id'] != u_identifier):
        return "Error: Unauthorized access"
    
    # Check if both accounts exist
    if account_from not in ACCOUNTS_DB or account_to not in ACCOUNTS_DB:
        return "Error: Invalid account number"
    
    # Check if sufficient balance
    if ACCOUNTS_DB[account_from]['balance'] < transfer_amount:
        return "Error: Insufficient funds"
    
    # Check if transfer amount is valid
    if transfer_amount <= 0:
        return "Error: Invalid transfer amount"
    
    # Perform the transfer
    ACCOUNTS_DB[account_from]['balance'] -= transfer_amount
    ACCOUNTS_DB[account_to]['balance'] += transfer_amount
    
    return f"Success: Transferred ${transfer_amount:.2f} from {account_from} to {account_to}"
