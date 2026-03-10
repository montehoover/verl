def authenticate_user(user_id: str, password: str) -> bool:
    """
    Authenticates a user based on their user_id and password.

    Args:
        user_id: The user's identifier.
        password: The user's password.

    Returns:
        True if the credentials are valid, False otherwise.
    """
    # This is a placeholder for actual authentication logic.
    # In a real application, you would check credentials against a database
    # or another secure authentication mechanism.
    # For demonstration purposes, let's assume a hardcoded valid user.
    if user_id == "admin" and password == "password123":
        return True
    return False

# Placeholder for account balances. In a real application, this would be a database.
account_balances = {
    "1234567890": 1000.00,
    "0987654321": 500.50,
}

def validate_and_update_balance(account_number: str, amount: float) -> bool:
    """
    Validates if the account has sufficient funds and updates the balance.
    A positive amount indicates a deposit, a negative amount indicates a withdrawal.

    Args:
        account_number: The account number.
        amount: The amount to transact. Positive for deposit, negative for withdrawal.

    Returns:
        True if the transaction is successful, False otherwise.
    """
    if account_number not in account_balances:
        print(f"Account {account_number} not found.")
        return False

    current_balance = account_balances[account_number]

    if amount < 0:  # Withdrawal
        if current_balance + amount < 0:
            print(f"Insufficient funds in account {account_number} for withdrawal of {-amount}.")
            return False
    
    # Update balance
    account_balances[account_number] += amount
    print(f"Transaction successful. New balance for account {account_number}: {account_balances[account_number]:.2f}")
    return True
