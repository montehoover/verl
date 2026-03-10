# Assume ACCOUNTS_DB is available
ACCOUNTS_DB = {}  # Example structure: {'12345': {'balance': 1000.50}, '67890': {'balance': 2500.75}}

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
