# Mock database for demonstration purposes
ACCOUNTS_DB = {}

def account_exists(account_number):
    """
    Check if a given account number exists in the database.
    
    Args:
        account_number (str): The account number to check
        
    Returns:
        bool: True if the account exists, False otherwise
    """
    return account_number in ACCOUNTS_DB

def get_balance(account_number):
    """
    Get the current balance of an account.
    
    Args:
        account_number (str): The account number to check
        
    Returns:
        float or None: The account balance if the account exists, None otherwise
    """
    if account_number in ACCOUNTS_DB:
        return ACCOUNTS_DB[account_number]
    return None
