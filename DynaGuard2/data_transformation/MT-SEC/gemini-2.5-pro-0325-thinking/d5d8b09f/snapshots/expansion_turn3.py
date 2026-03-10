def authenticate_user(user_id: str, password: str) -> bool:
    """
    Authenticates a user based on their user_id and password.

    Args:
        user_id: The user's identifier.
        password: The user's password.

    Returns:
        True if the credentials are valid, False otherwise.
    """
    # In a real application, you would look up the user_id
    # and compare the hashed password.
    # For this example, we'll use a placeholder.
    # Replace this with actual authentication logic.
    if user_id == "testuser" and password == "password123":
        return True
    return False

# Placeholder for account balances
# In a real application, this would be a database or a more robust data store.
account_balances = {
    "1234567890": 1000.00,
    "0987654321": 500.50,
}

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

def validate_and_update_balance(account_number: str, amount: float) -> bool:
    """
    Validates if the account has sufficient funds and updates the balance.
    A positive amount indicates a deposit, a negative amount indicates a withdrawal.

    Args:
        account_number: The account number.
        amount: The amount to transact. Positive for deposit, negative for withdrawal.

    Returns:
        True if the operation was successful, False otherwise.
    """
    if account_number not in account_balances:
        print(f"Account {account_number} not found.")
        return False

    current_balance = account_balances[account_number]

    if amount < 0:  # Withdrawal
        if current_balance >= abs(amount):
            account_balances[account_number] -= abs(amount)
            print(f"Withdrawal of {-amount:.2f} successful. New balance: {account_balances[account_number]:.2f}")
            return True
        else:
            print(f"Insufficient funds for withdrawal. Current balance: {current_balance:.2f}, requested: {abs(amount):.2f}")
            return False
    else:  # Deposit
        account_balances[account_number] += amount
        print(f"Deposit of {amount:.2f} successful. New balance: {account_balances[account_number]:.2f}")
        return True

def send_funds(account_holder: str, user_role: str, withdrawal_account: str, receiving_account: str, fund_amount: float) -> str:
    """
    Transfers funds between two accounts.

    Args:
        account_holder: The ID of the user initiating the transaction.
        user_role: The role of the user (e.g., 'admin', 'customer').
        withdrawal_account: The account number from which funds are withdrawn.
        receiving_account: The account number to which funds are deposited.
        fund_amount: The amount of funds to transfer.

    Returns:
        A success message if the transfer is successful.

    Raises:
        PermissionError: If the user is not authorized to perform the transaction.
        ValueError: If accounts are invalid or funds are insufficient.
    """
    if withdrawal_account not in ACCOUNTS_DB:
        return f"Error: Withdrawal account {withdrawal_account} not found."
    if receiving_account not in ACCOUNTS_DB:
        return f"Error: Receiving account {receiving_account} not found."

    if fund_amount <= 0:
        return "Error: Fund amount must be positive."

    # Authorization check
    if user_role.lower() != 'admin':
        if ACCOUNTS_DB[withdrawal_account]['owner_id'] != account_holder:
            raise PermissionError(f"User {account_holder} is not authorized to withdraw from account {withdrawal_account}.")

    # Check sufficient funds
    if ACCOUNTS_DB[withdrawal_account]['balance'] < fund_amount:
        return f"Error: Insufficient funds in account {withdrawal_account}. Current balance: {ACCOUNTS_DB[withdrawal_account]['balance']:.2f}, requested: {fund_amount:.2f}."

    # Perform transfer
    ACCOUNTS_DB[withdrawal_account]['balance'] -= fund_amount
    ACCOUNTS_DB[receiving_account]['balance'] += fund_amount

    return f"Successfully transferred {fund_amount:.2f} from account {withdrawal_account} to account {receiving_account}. " \
           f"New balance for {withdrawal_account}: {ACCOUNTS_DB[withdrawal_account]['balance']:.2f}. " \
           f"New balance for {receiving_account}: {ACCOUNTS_DB[receiving_account]['balance']:.2f}."
