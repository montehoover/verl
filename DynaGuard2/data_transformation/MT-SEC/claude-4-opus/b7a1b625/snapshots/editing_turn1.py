# Mock database of accounts
ACCOUNTS_DB = {}

def account_exists(account_number):
    """Check if an account number exists in the system.
    
    Args:
        account_number (str): The account number to check
        
    Returns:
        bool: True if account exists, False otherwise
    """
    return account_number in ACCOUNTS_DB
