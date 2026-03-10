# Sample ACCOUNTS_DB for demonstration purposes
ACCOUNTS_DB = {
    "123456": {"owner_id": "user001", "balance": 1500.00},
    "789012": {"owner_id": "user002", "balance": 3200.50},
    "345678": {"owner_id": "user003", "balance": 750.25}
}

def get_account_details(account_number):
    """
    Retrieves account details for a given account number.
    
    Args:
        account_number (str): The account number to look up
        
    Returns:
        dict: Account details including owner_id and balance, or None if not found
    """
    if account_number in ACCOUNTS_DB:
        return ACCOUNTS_DB[account_number]
    return None
