# Placeholder for account data. In a real application, this would be a database.
ACCOUNTS_DB = {
    "1234567890": {"balance": 1000.00},
    "0987654321": {"balance": 500.00},
    "1122334455": {"balance": 15000.00},
}

def authenticate_user(user_id: str, password: str) -> bool:
    """
    Authenticates a user based on their user_id and password.

    Args:
        user_id: The user's identifier.
        password: The user's password.

    Returns:
        True if the credentials are correct, False otherwise.
    """
    # In a real application, you would look up the user_id in a database
    # and compare the hashed password.
    # For this example, we'll use a placeholder.
    # Replace this with actual authentication logic.
    if user_id == "admin" and password == "password123":
        return True
    return False

def validate_and_update_balance(account_number: str, amount: float) -> bool:
    """
    Validates if a transaction is possible and updates the account balance.

    Args:
        account_number: The account number to update.
        amount: The amount to add (positive for deposit, negative for withdrawal).

    Returns:
        True if the balance was updated successfully, False otherwise.
    """
    if account_number not in ACCOUNTS_DB:
        return False  # Account does not exist

    current_balance = ACCOUNTS_DB[account_number]["balance"]

    # Check for sufficient funds only if it's a withdrawal (amount is negative)
    if amount < 0 and current_balance + amount < 0:
        return False  # Insufficient funds for withdrawal

    # Update balance
    ACCOUNTS_DB[account_number]["balance"] += amount
    return True
