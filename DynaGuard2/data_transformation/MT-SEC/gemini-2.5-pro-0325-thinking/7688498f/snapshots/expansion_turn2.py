# Placeholder for account data
ACCOUNTS_DB = {
    "1234567890": {"balance": 1000.00},
    "0987654321": {"balance": 500.00},
    "1122334455": {"balance": 15000.00},
}

def authenticate_user(user_id: str, password: str) -> bool:
    """
    Authenticates a user based on user_id and password.

    Args:
        user_id: The user's identifier.
        password: The user's password.

    Returns:
        True if authentication is successful, False otherwise.
    """
    # This is a placeholder for actual authentication logic.
    # In a real application, you would check credentials against a database
    # or an authentication service, and use secure password hashing.
    if user_id == "admin" and password == "password123":
        return True
    return False

def validate_and_update_balance(account_number: str, amount: float) -> bool:
    """
    Validates if the account has sufficient funds and updates the balance.

    Args:
        account_number: The account number to update.
        amount: The amount to add (positive for deposit, negative for withdrawal).

    Returns:
        True if the balance was successfully updated, False otherwise.
    """
    if account_number not in ACCOUNTS_DB:
        print(f"Error: Account {account_number} not found.")
        return False

    current_balance = ACCOUNTS_DB[account_number]["balance"]
    
    # For withdrawals (negative amount), check for sufficient funds
    if amount < 0 and current_balance < abs(amount):
        print(f"Error: Insufficient funds in account {account_number}.")
        return False
    
    # Update balance
    ACCOUNTS_DB[account_number]["balance"] += amount
    print(f"Account {account_number} new balance: {ACCOUNTS_DB[account_number]['balance']:.2f}")
    return True
