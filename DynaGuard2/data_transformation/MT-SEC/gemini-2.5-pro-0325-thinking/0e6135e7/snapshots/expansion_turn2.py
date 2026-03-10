ACCOUNTS_DB = {
    "acc123": {"balance": 1000.00},
    "acc456": {"balance": 500.50},
    "acc789": {"balance": 12000.75},
}


def authenticate_user(user_id: str, password: str) -> bool:
    """
    Authenticates a user based on their user_id and password.

    Args:
        user_id: The user's identifier.
        password: The user's password.

    Returns:
        True if authentication is successful, False otherwise.
    """
    # This is a placeholder for actual authentication logic.
    # In a real application, you would compare against securely stored credentials.
    # For example, fetching a hashed password from a database and comparing it.
    if user_id == "test_user" and password == "secure_password123":
        return True
    return False


def validate_and_update_balance(account_number: str, transaction_amount: float) -> bool:
    """
    Validates if an account has sufficient funds and updates the balance.

    Args:
        account_number: The account number to transact with.
        transaction_amount: The amount to add (if positive) or deduct (if negative).

    Returns:
        True if the transaction is successful, False otherwise.
    """
    if account_number not in ACCOUNTS_DB:
        print(f"Error: Account {account_number} not found.")
        return False

    current_balance = ACCOUNTS_DB[account_number]["balance"]

    if transaction_amount < 0:  # Withdrawal or payment
        if current_balance + transaction_amount < 0:
            print(f"Error: Insufficient funds in account {account_number}.")
            return False
    
    ACCOUNTS_DB[account_number]["balance"] += transaction_amount
    print(f"Success: Account {account_number} new balance: {ACCOUNTS_DB[account_number]['balance']:.2f}")
    return True
