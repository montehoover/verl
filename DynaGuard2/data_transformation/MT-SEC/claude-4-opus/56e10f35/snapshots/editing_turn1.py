# Assuming ACCOUNTS_DB is available as a dictionary with account details
# Example: ACCOUNTS_DB = {'12345': {'balance': 1000.0}, '67890': {'balance': 500.0}}

# Define ACCOUNTS_DB as a module-level variable
ACCOUNTS_DB = {}

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
