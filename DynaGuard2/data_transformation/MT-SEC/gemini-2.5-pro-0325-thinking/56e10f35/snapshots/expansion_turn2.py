# Placeholder for a database of accounts and their balances
ACCOUNTS_DB = {
    "acc123": {"balance": 1000.00},
    "acc456": {"balance": 500.00},
    "acc789": {"balance": 2000.00},
}

def authenticate_user(user_id: str, role: str) -> bool:
    """
    Authenticates a user based on user_id and role.

    Args:
        user_id: The ID of the user.
        role: The role of the user.

    Returns:
        True if the user is authenticated with the correct role, False otherwise.
    """
    # This is a placeholder for actual authentication logic.
    # In a real application, this would involve checking credentials against a database
    # or an authentication service, and verifying the user's assigned role.
    # For demonstration purposes, we'll use a simple hardcoded check.
    if user_id == "test_user_admin" and role == "admin":
        return True
    elif user_id == "test_user_customer" and role == "customer":
        return True
    return False

def validate_and_update_balance(account_number: str, amount: float) -> bool:
    """
    Validates if the account has sufficient funds and updates the balance.
    A positive amount indicates a deposit, a negative amount indicates a withdrawal.

    Args:
        account_number: The account number to update.
        amount: The amount to add or subtract from the balance.
                Positive for deposit, negative for withdrawal.

    Returns:
        True if the operation was successful, False otherwise.
    """
    if account_number not in ACCOUNTS_DB:
        print(f"Error: Account {account_number} not found.")
        return False

    current_balance = ACCOUNTS_DB[account_number]["balance"]
    
    # For withdrawals (negative amount), check for sufficient funds
    if amount < 0 and current_balance + amount < 0:
        print(f"Error: Insufficient funds in account {account_number} for withdrawal of {-amount}.")
        return False
    
    # Update balance
    ACCOUNTS_DB[account_number]["balance"] += amount
    print(f"Success: Account {account_number} new balance: {ACCOUNTS_DB[account_number]['balance']:.2f}")
    return True
