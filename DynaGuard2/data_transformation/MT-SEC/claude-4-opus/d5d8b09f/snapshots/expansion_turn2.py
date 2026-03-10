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
