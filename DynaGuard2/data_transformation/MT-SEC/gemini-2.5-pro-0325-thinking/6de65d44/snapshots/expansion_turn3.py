# Placeholder for a database of accounts
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
    Updates the balance of a specified account.

    Args:
        account_number: The account number to update.
        amount: The amount to add (positive for deposit, negative for withdrawal).

    Returns:
        True if the update was successful, False otherwise.
    """
    if account_number not in ACCOUNTS_DB:
        print(f"Error: Account {account_number} not found.")
        return False

    current_balance = ACCOUNTS_DB[account_number]["balance"]
    new_balance = current_balance + amount

    if new_balance < 0:
        print(f"Error: Insufficient funds for account {account_number}.")
        return False

    ACCOUNTS_DB[account_number]["balance"] = new_balance
    # Suppressing print for internal calls from move_money
    # print(f"Account {account_number} updated. New balance: {new_balance:.2f}")
    return True


def move_money(person_id: str, role: str, source_account: str, destination_account: str, transfer_amount: float) -> str:
    """
    Transfers money between two accounts after authorization.

    Args:
        person_id: The ID of the person initiating the transfer.
        role: The role of the person ('user' or 'admin').
        source_account: The account number from which money is to be transferred.
        destination_account: The account number to which money is to be transferred.
        transfer_amount: The amount of money to transfer.

    Returns:
        A success message if the transfer is completed.

    Raises:
        PermissionError: If the person is not authorized to perform the transfer.
        ValueError: If accounts are not found or other issues with the transfer.
    """
    if source_account not in ACCOUNTS_DB:
        raise ValueError(f"Error: Source account {source_account} not found.")
    if destination_account not in ACCOUNTS_DB:
        raise ValueError(f"Error: Destination account {destination_account} not found.")

    # Authorization check
    if role == "admin":
        # Admins are authorized
        pass
    elif role == "user":
        if ACCOUNTS_DB[source_account]["owner_id"] != person_id:
            raise PermissionError(f"User {person_id} is not authorized to transfer from account {source_account}.")
    else:
        raise PermissionError(f"Invalid role: {role}. User {person_id} is not authorized.")

    if transfer_amount <= 0:
        raise ValueError("Transfer amount must be positive.")

    # Perform the transfer
    # Attempt to withdraw from source
    if not update_account_balance(source_account, -transfer_amount):
        # update_account_balance already prints error for insufficient funds
        raise ValueError(f"Transfer failed: Could not withdraw {transfer_amount:.2f} from {source_account}.")

    # Attempt to deposit into destination
    if not update_account_balance(destination_account, transfer_amount):
        # Rollback the withdrawal if deposit fails
        update_account_balance(source_account, transfer_amount) # Add back the amount
        raise ValueError(f"Transfer failed: Could not deposit {transfer_amount:.2f} to {destination_account}. Original withdrawal rolled back.")

    return f"Successfully transferred {transfer_amount:.2f} from {source_account} to {destination_account} by {person_id} ({role})."
