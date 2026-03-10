# A mock database for account balances
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


def process_transfer(uid: str, role: str, source_account: str, destination_account: str, transfer_amount: float) -> str:
    """
    Processes a money transfer between two accounts.

    Args:
        uid: The user ID initiating the transfer.
        role: The role of the user ('admin' or 'user').
        source_account: The account number from which money is to be transferred.
        destination_account: The account number to which money is to be transferred.
        transfer_amount: The amount of money to transfer.

    Returns:
        A success message if the transfer is successful.

    Raises:
        PermissionError: If the user is not authorized to perform the transfer.
        ValueError: If the transfer amount is not positive.
    """
    if transfer_amount <= 0:
        return "Error: Transfer amount must be positive."

    if source_account not in ACCOUNTS_DB:
        return f"Error: Source account {source_account} not found."
    if destination_account not in ACCOUNTS_DB:
        return f"Error: Destination account {destination_account} not found."

    # Authorization check
    source_account_details = ACCOUNTS_DB[source_account]
    is_authorized = False
    if role == "admin":
        is_authorized = True
    elif role == "user" and source_account_details["owner_id"] == uid:
        is_authorized = True

    if not is_authorized:
        raise PermissionError(
            f"User {uid} with role {role} is not authorized to transfer from account {source_account}."
        )

    # Check for sufficient funds
    if source_account_details["balance"] < transfer_amount:
        return f"Error: Insufficient funds in source account {source_account}. " \
               f"Available: {source_account_details['balance']:.2f}, Requested: {transfer_amount:.2f}."

    # Perform the transfer
    ACCOUNTS_DB[source_account]["balance"] -= transfer_amount
    ACCOUNTS_DB[destination_account]["balance"] += transfer_amount

    return f"Success: Transferred {transfer_amount:.2f} from {source_account} to {destination_account}. " \
           f"New balance for {source_account}: {ACCOUNTS_DB[source_account]['balance']:.2f}. " \
           f"New balance for {destination_account}: {ACCOUNTS_DB[destination_account]['balance']:.2f}."
