# Assume ACCOUNTS_DB is available with structure like:
# ACCOUNTS_DB = {
#     "12345": {"balance": 1000.00},
#     "67890": {"balance": 500.00}
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
