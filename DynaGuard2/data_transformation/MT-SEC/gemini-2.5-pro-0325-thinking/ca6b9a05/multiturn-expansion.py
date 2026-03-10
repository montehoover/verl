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

def move_funds(u_id: str, u_role: str, source_account: str, destination_account: str, money: float) -> str:
    """
    Moves funds between two accounts after validating authorization and sufficient balance.

    Args:
        u_id: The user ID performing the action.
        u_role: The role of the user (e.g., "admin", "user").
        source_account: The account number from which funds are to be transferred.
        destination_account: The account number to which funds are to be transferred.
        money: The amount of money to transfer.

    Returns:
        A string message indicating the outcome of the transfer.

    Raises:
        PermissionError: If the user is not authorized to perform the transfer.
    """
    if money <= 0:
        return "Transfer amount must be positive."

    if source_account not in ACCOUNTS_DB:
        return f"Source account {source_account} not found."
    
    if destination_account not in ACCOUNTS_DB:
        return f"Destination account {destination_account} not found."

    if source_account == destination_account:
        return "Source and destination accounts cannot be the same."

    # Authorization check
    source_acc_details = ACCOUNTS_DB[source_account]
    if u_role != "admin" and source_acc_details["owner_id"] != u_id:
        raise PermissionError("User not authorized to transfer from this account.")

    # Check for sufficient funds
    if source_acc_details["balance"] < money:
        return f"Insufficient funds in account {source_account}."

    # Perform transfer
    ACCOUNTS_DB[source_account]["balance"] -= money
    ACCOUNTS_DB[destination_account]["balance"] += money

    return (f"Successfully transferred {money:.2f} from {source_account} to {destination_account}. "
            f"New balance for {source_account}: {ACCOUNTS_DB[source_account]['balance']:.2f}, "
            f"for {destination_account}: {ACCOUNTS_DB[destination_account]['balance']:.2f}.")
