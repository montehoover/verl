# Assume ACCOUNTS_DB is available with structure like:
# ACCOUNTS_DB = {
#     "12345": {"balance": 1000.00, "owner_id": "user123"},
#     "67890": {"balance": 500.00, "owner_id": "user456"}
# }

# Example ACCOUNTS_DB for testing
ACCOUNTS_DB = {}

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
