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


def authenticate_user(user_id: str, password: str) -> bool:
    """
    Authenticate a user based on their user_id and password.
    
    Args:
        user_id (str): The user's ID
        password (str): The user's password
        
    Returns:
        bool: True if credentials are valid, False otherwise
    """
    # This is a placeholder implementation
    # In a real application, you would:
    # 1. Hash the password
    # 2. Query a database for the user
    # 3. Compare the hashed password with stored hash
    # 4. Implement proper security measures
    
    # For now, this is a simple mock implementation
    # DO NOT use this in production
    mock_users = {
        "user123": "password123",
        "admin": "admin_pass",
        "john_doe": "secure_password"
    }
    
    return mock_users.get(user_id) == password


def validate_and_update_balance(account_number: str, amount: float) -> bool:
    """
    Validate if an account has sufficient funds and update the balance.
    
    Args:
        account_number (str): The account number
        amount (float): The amount to deduct (negative) or add (positive)
        
    Returns:
        bool: True if operation was successful, False otherwise
    """
    # Mock account balances
    # In a real application, this would be stored in a database
    mock_accounts = {
        "ACC001": 1000.00,
        "ACC002": 5000.00,
        "ACC003": 250.00,
        "ACC004": 10000.00
    }
    
    # Check if account exists
    if account_number not in mock_accounts:
        return False
    
    current_balance = mock_accounts[account_number]
    new_balance = current_balance + amount
    
    # Check if withdrawal would result in negative balance
    if new_balance < 0:
        return False
    
    # Update the balance
    mock_accounts[account_number] = new_balance
    return True


def send_funds(account_holder: str, user_role: str, withdrawal_account: str, receiving_account: str, fund_amount: float) -> str:
    """
    Transfer funds between accounts with permission checking.
    
    Args:
        account_holder (str): The user ID of the person initiating the transfer
        user_role (str): The role of the user (e.g., 'owner', 'admin')
        withdrawal_account (str): The account to withdraw from
        receiving_account (str): The account to deposit to
        fund_amount (float): The amount to transfer
        
    Returns:
        str: Success message
        
    Raises:
        PermissionError: If user is not authorized to perform the transfer
    """
    # Check if withdrawal account exists
    if withdrawal_account not in ACCOUNTS_DB:
        raise PermissionError("Withdrawal account does not exist")
    
    # Check if receiving account exists
    if receiving_account not in ACCOUNTS_DB:
        raise PermissionError("Receiving account does not exist")
    
    # Check permissions
    withdrawal_account_data = ACCOUNTS_DB[withdrawal_account]
    if user_role != "admin" and withdrawal_account_data["owner_id"] != account_holder:
        raise PermissionError("You do not have permission to transfer funds from this account")
    
    # Check if withdrawal account has sufficient funds
    if withdrawal_account_data["balance"] < fund_amount:
        raise PermissionError("Insufficient funds in withdrawal account")
    
    # Perform the transfer
    ACCOUNTS_DB[withdrawal_account]["balance"] -= fund_amount
    ACCOUNTS_DB[receiving_account]["balance"] += fund_amount
    
    return f"Successfully transferred ${fund_amount:.2f} from {withdrawal_account} to {receiving_account}"
