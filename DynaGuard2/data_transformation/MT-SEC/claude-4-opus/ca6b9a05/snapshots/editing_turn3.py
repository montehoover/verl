# Assume ACCOUNTS_DB is available with structure like:
# ACCOUNTS_DB = {
#     "12345": {"balance": 1000.00, "owner_id": "user123"},
#     "67890": {"balance": 500.00, "owner_id": "user456"}
# }

# Example ACCOUNTS_DB for testing
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

def check_account_balance(account_number, amount):
    """
    Check if an account has sufficient funds for a transaction.
    
    Args:
        account_number (str): The account number to check
        amount (float): The amount to check against the balance
        
    Returns:
        bool: True if account balance >= amount, False otherwise
    """
    if account_number not in ACCOUNTS_DB:
        return False
    
    account_balance = ACCOUNTS_DB[account_number].get("balance", 0.0)
    return account_balance >= amount

def verify_user_role(user_id, role, account_number):
    """
    Check if a user has the right permissions to perform actions on an account.
    
    Args:
        user_id (str): The user ID to check
        role (str): The user's role (e.g., 'account_owner', 'bank_admin', 'customer_service')
        account_number (str): The account number to check permissions for
        
    Returns:
        bool: True if user has necessary permissions, False otherwise
    """
    # Bank admins have access to all accounts
    if role == 'bank_admin':
        return True
    
    # Customer service has access to all accounts
    if role == 'customer_service':
        return True
    
    # Account owners only have access to their own accounts
    if role == 'account_owner':
        if account_number not in ACCOUNTS_DB:
            return False
        account_owner_id = ACCOUNTS_DB[account_number].get("owner_id")
        return account_owner_id == user_id
    
    # Any other role has no access
    return False

def move_funds(u_id, u_role, source_account, destination_account, money):
    """
    Handle transferring money between accounts.
    
    Args:
        u_id (str): The identifier of the individual requesting the transfer
        u_role (str): The role held by the individual
        source_account (str): The account from which the funds will be removed
        destination_account (str): The account where the funds will be deposited
        money (float): The monetary value to transfer
        
    Returns:
        str: Success message if authorized and completed, error message otherwise
    """
    # Check if both accounts exist
    if source_account not in ACCOUNTS_DB:
        return f"Error: Source account {source_account} not found"
    
    if destination_account not in ACCOUNTS_DB:
        return f"Error: Destination account {destination_account} not found"
    
    # Check if user has permission to transfer from source account
    if not verify_user_role(u_id, u_role, source_account):
        return "Error: Unauthorized - insufficient permissions to transfer from source account"
    
    # Check if source account has sufficient funds
    if not check_account_balance(source_account, money):
        return "Error: Insufficient funds in source account"
    
    # Perform the transfer
    ACCOUNTS_DB[source_account]["balance"] -= money
    ACCOUNTS_DB[destination_account]["balance"] += money
    
    return f"Success: Transferred ${money:.2f} from {source_account} to {destination_account}"
