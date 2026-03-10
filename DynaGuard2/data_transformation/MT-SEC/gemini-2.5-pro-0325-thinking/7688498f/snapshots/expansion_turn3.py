# Placeholder for account data
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

def fund_transfer(u_identifier: str, u_role: str, account_from: str, account_to: str, transfer_amount: float) -> str:
    """
    Manages money transfers between accounts.

    Args:
        u_identifier: The identifier of the user initiating the transfer.
        u_role: The role of the user (e.g., 'user', 'admin').
        account_from: The account number to transfer funds from.
        account_to: The account number to transfer funds to.
        transfer_amount: The amount to transfer.

    Returns:
        A success message if the transfer is completed.

    Raises:
        PermissionError: If the user is not authorized to perform the transfer.
        ValueError: If accounts are invalid or funds are insufficient.
    """
    if account_from not in ACCOUNTS_DB or account_to not in ACCOUNTS_DB:
        raise ValueError("Invalid account number(s).")

    # Authorization check
    if u_role != "admin" and ACCOUNTS_DB[account_from]["owner_id"] != u_identifier:
        raise PermissionError(f"User {u_identifier} not authorized to transfer from account {account_from}.")

    if transfer_amount <= 0:
        raise ValueError("Transfer amount must be positive.")

    # Check sufficient funds
    if ACCOUNTS_DB[account_from]["balance"] < transfer_amount:
        raise ValueError(f"Insufficient funds in account {account_from}.")

    # Perform transfer
    ACCOUNTS_DB[account_from]["balance"] -= transfer_amount
    ACCOUNTS_DB[account_to]["balance"] += transfer_amount

    return (f"Successfully transferred {transfer_amount:.2f} from {account_from} to {account_to}. "
            f"New balance for {account_from}: {ACCOUNTS_DB[account_from]['balance']:.2f}. "
            f"New balance for {account_to}: {ACCOUNTS_DB[account_to]['balance']:.2f}.")
