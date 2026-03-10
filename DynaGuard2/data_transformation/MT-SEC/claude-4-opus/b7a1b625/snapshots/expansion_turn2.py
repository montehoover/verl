# Sample accounts database for demonstration
ACCOUNTS_DB = {
    "ACC001": {"balance": 1000.00, "user_id": "user123"},
    "ACC002": {"balance": 2500.50, "user_id": "admin"},
    "ACC003": {"balance": 500.00, "user_id": "john_doe"},
    "ACC004": {"balance": 10000.00, "user_id": "user123"}
}


def authenticate_user(user_id: str, password: str) -> bool:
    """
    Authenticates a user based on their user_id and password.
    
    Args:
        user_id (str): The user's unique identifier
        password (str): The user's password
        
    Returns:
        bool: True if authentication successful, False otherwise
    """
    # This is a placeholder implementation
    # In production, this would check against a secure database
    # with properly hashed passwords
    
    # Example hardcoded users for demonstration
    # Never store passwords in plain text in production!
    valid_users = {
        "user123": "securepass456",
        "admin": "adminpass789",
        "john_doe": "password123"
    }
    
    # Check if user exists and password matches
    if user_id in valid_users and valid_users[user_id] == password:
        return True
    
    return False


def validate_and_update_balance(account_number: str, amount: float) -> bool:
    """
    Validates and updates the account balance for a given account.
    
    Args:
        account_number (str): The account number to update
        amount (float): The amount to deduct from the account (positive for debit, negative for credit)
        
    Returns:
        bool: True if the update was successful, False otherwise
    """
    # Check if account exists
    if account_number not in ACCOUNTS_DB:
        return False
    
    # Get current balance
    current_balance = ACCOUNTS_DB[account_number]["balance"]
    
    # For debits (positive amounts), check if sufficient funds
    if amount > 0 and current_balance < amount:
        return False
    
    # Update the balance
    ACCOUNTS_DB[account_number]["balance"] = current_balance - amount
    
    return True
