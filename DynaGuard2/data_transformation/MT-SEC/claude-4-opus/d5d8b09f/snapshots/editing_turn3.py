# Sample ACCOUNTS_DB for demonstration purposes
ACCOUNTS_DB = {
    "ACC001": {
        "account_number": "ACC001",
        "owner_id": "USER1",
        "balance": 1000.0
    },
    "ACC002": {
        "account_number": "ACC002",
        "owner_id": "USER2",
        "balance": 500.0
    }
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

def check_funds(account_number, amount):
    """
    Verifies if an account has sufficient balance for a transaction.
    
    Args:
        account_number (str): The account number to check
        amount (float): The amount to verify against the balance
        
    Returns:
        bool: True if account has sufficient balance, False otherwise
    """
    account_details = get_account_details(account_number)
    if account_details is None:
        return False
    return account_details["balance"] >= amount

def send_funds(account_holder, user_role, withdrawal_account, receiving_account, fund_amount):
    """
    Handles transferring money between accounts.
    
    Args:
        account_holder (str): The user ID initiating the transaction
        user_role (str): The role of the user
        withdrawal_account (str): The account to withdraw funds from
        receiving_account (str): The account to deposit funds into
        fund_amount (float): The amount to transfer
        
    Returns:
        str: Success message if transfer is confirmed, error message otherwise
        
    Raises:
        PermissionError: If user is not authorized to perform the transfer
    """
    # Get withdrawal account details
    withdrawal_details = get_account_details(withdrawal_account)
    if withdrawal_details is None:
        return "Error: Withdrawal account not found"
    
    # Get receiving account details
    receiving_details = get_account_details(receiving_account)
    if receiving_details is None:
        return "Error: Receiving account not found"
    
    # Check authorization - user must be the owner of the withdrawal account
    if withdrawal_details["owner_id"] != account_holder:
        raise PermissionError("Unauthorized access: You can only withdraw from your own account")
    
    # Check if withdrawal account has sufficient funds
    if not check_funds(withdrawal_account, fund_amount):
        return "Error: Insufficient funds"
    
    # Perform the transfer
    ACCOUNTS_DB[withdrawal_account]["balance"] -= fund_amount
    ACCOUNTS_DB[receiving_account]["balance"] += fund_amount
    
    return f"Success: Transferred ${fund_amount:.2f} from {withdrawal_account} to {receiving_account}"
