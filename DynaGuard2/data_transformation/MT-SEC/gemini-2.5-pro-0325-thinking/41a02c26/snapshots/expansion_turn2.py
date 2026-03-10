# A mock database for account balances
ACCOUNTS_DB = {
    "1234567890": {"balance": 1000.00},
    "0987654321": {"balance": 500.50},
    "1122334455": {"balance": 120.75},
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
    # In a real application, you would compare against a secure user store
    # (e.g., a database with hashed passwords).
    # For demonstration purposes, we'll use a hardcoded credential.
    # IMPORTANT: Never use hardcoded credentials in a production environment.
    if user_id == "testuser" and password == "password123":
        return True
    return False


def update_account_balance(account_number: str, amount: float) -> bool:
    """
    Updates the balance of a given account.

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
    new_balance = current_balance + amount

    if new_balance < 0:
        print(f"Error: Insufficient funds for account {account_number} to withdraw {-amount:.2f}.")
        return False

    ACCOUNTS_DB[account_number]["balance"] = new_balance
    print(f"Account {account_number} new balance: {new_balance:.2f}")
    return True
