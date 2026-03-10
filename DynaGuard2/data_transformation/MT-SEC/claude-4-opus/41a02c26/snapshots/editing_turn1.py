# Assume ACCOUNTS_DB is available with account details
# Example: ACCOUNTS_DB = {'123456': {'balance': 1000.00}, '789012': {'balance': 500.00}}

# Mock ACCOUNTS_DB for testing/demonstration
ACCOUNTS_DB = {}

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
