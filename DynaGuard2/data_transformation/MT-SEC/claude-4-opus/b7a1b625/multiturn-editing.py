# Mock database of accounts
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

def account_exists(account_number):
    """Check if an account number exists in the system.
    
    Args:
        account_number (str): The account number to check
        
    Returns:
        bool: True if account exists, False otherwise
    """
    return account_number in ACCOUNTS_DB

def get_balance(account_number):
    """Get the balance of an account.
    
    Args:
        account_number (str): The account number to check
        
    Returns:
        float or None: The account balance if exists, None otherwise
    """
    account = ACCOUNTS_DB.get(account_number)
    if account:
        return account['balance']
    return None

def execute_transfer(employee_id, role_value, source_acct, target_acct, total_amount):
    """Execute a money transfer between accounts.
    
    Args:
        employee_id (str): ID of the user initiating the transfer
        role_value (str): The user's role
        source_acct (str): The account number to withdraw from
        target_acct (str): The account to deposit into
        total_amount (float): The amount to transfer
        
    Returns:
        str: Success message if authorized, error message otherwise
    """
    # Check if both accounts exist
    if not account_exists(source_acct):
        return f"Error: Source account {source_acct} does not exist"
    if not account_exists(target_acct):
        return f"Error: Target account {target_acct} does not exist"
    
    # Check if source account has sufficient balance
    source_balance = get_balance(source_acct)
    if source_balance < total_amount:
        return f"Error: Insufficient funds in source account"
    
    # Check authorization (simple role-based check)
    authorized_roles = ["admin", "teller", "manager"]
    if role_value.lower() not in authorized_roles:
        return f"Error: User with role '{role_value}' is not authorized to perform transfers"
    
    # Perform the transfer
    ACCOUNTS_DB[source_acct]['balance'] -= total_amount
    ACCOUNTS_DB[target_acct]['balance'] += total_amount
    
    return f"Transfer successful: {total_amount} transferred from {source_acct} to {target_acct} by employee {employee_id}"
