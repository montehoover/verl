# Assume ACCOUNTS_DB is defined elsewhere in the application
# For example: ACCOUNTS_DB = {'12345': {'balance': 1000.0}, '67890': {'balance': 500.0}}

# Mock database for accounts (this would normally be imported from another module)
ACCOUNTS_DB = {}

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
