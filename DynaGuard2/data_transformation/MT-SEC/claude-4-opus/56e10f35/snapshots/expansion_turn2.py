def authenticate_user(user_id: str, role: str) -> bool:
    """
    Authenticate a user based on their user_id and role.
    
    Args:
        user_id (str): The unique identifier of the user
        role (str): The role to check for the user (e.g., 'admin', 'customer', 'teller')
    
    Returns:
        bool: True if the user is authenticated with the correct role, False otherwise
    """
    # Mock user database with user_id: role mappings
    # In a real application, this would query a secure database
    user_roles = {
        "user001": "admin",
        "user002": "customer",
        "user003": "teller",
        "user004": "customer",
        "user005": "admin"
    }
    
    # Check if user exists and has the correct role
    if user_id in user_roles and user_roles[user_id] == role:
        return True
    
    return False


# Mock accounts database
# In a real application, this would be a secure database
ACCOUNTS_DB = {
    "ACC001": {"balance": 1000.00},
    "ACC002": {"balance": 2500.50},
    "ACC003": {"balance": 500.00},
    "ACC004": {"balance": 10000.00},
    "ACC005": {"balance": 0.00}
}


def validate_and_update_balance(account_number: str, amount: float) -> bool:
    """
    Validate if an account has sufficient funds and update the balance.
    
    Args:
        account_number (str): The account number to update
        amount (float): The amount to deduct from the account (negative for withdrawals, positive for deposits)
    
    Returns:
        bool: True if the operation was successful, False otherwise
    """
    # Check if account exists
    if account_number not in ACCOUNTS_DB:
        return False
    
    current_balance = ACCOUNTS_DB[account_number]["balance"]
    new_balance = current_balance + amount
    
    # Check if sufficient funds for withdrawal (negative amount)
    if new_balance < 0:
        return False
    
    # Update the balance
    ACCOUNTS_DB[account_number]["balance"] = new_balance
    return True
