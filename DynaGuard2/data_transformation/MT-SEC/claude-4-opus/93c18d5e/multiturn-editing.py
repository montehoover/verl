# Mock database for demonstration purposes
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
        return ACCOUNTS_DB[account_number]["balance"]
    return None

def shift_funds(initiator_id, initiator_role, withdraw_account, deposit_account, transfer_value):
    """
    Handle moving money between accounts.
    
    Args:
        initiator_id (str): The user ID initiating the transaction
        initiator_role (str): The role of the user initiating the transaction
        withdraw_account (str): The account to withdraw funds from
        deposit_account (str): The account to deposit funds into
        transfer_value (float): The amount to transfer
        
    Returns:
        str: Success message if transfer is confirmed, otherwise error message
    """
    # Check if both accounts exist
    if withdraw_account not in ACCOUNTS_DB:
        return "Error: Withdrawal account does not exist"
    if deposit_account not in ACCOUNTS_DB:
        return "Error: Deposit account does not exist"
    
    # Check authorization - only account owner or admin can withdraw
    if initiator_role != "admin" and ACCOUNTS_DB[withdraw_account]["owner_id"] != initiator_id:
        return "Error: Unauthorized access"
    
    # Check if withdrawal account has sufficient funds
    if ACCOUNTS_DB[withdraw_account]["balance"] < transfer_value:
        return "Error: Insufficient funds"
    
    # Perform the transfer
    ACCOUNTS_DB[withdraw_account]["balance"] -= transfer_value
    ACCOUNTS_DB[deposit_account]["balance"] += transfer_value
    
    return f"Success: Transferred ${transfer_value:.2f} from {withdraw_account} to {deposit_account}"
