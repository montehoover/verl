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

def get_balance(account_number):
    """Get the balance of an account.
    
    Args:
        account_number (str): The account number to check
        
    Returns:
        float or None: The account balance if exists, None otherwise
    """
    return ACCOUNTS_DB.get(account_number)
