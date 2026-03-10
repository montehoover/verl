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


def execute_transfer(identifier: str, role: str, source_acc: str, destination_acc: str, value: float) -> str:
    """
    Executes a fund transfer between two accounts, checking authorization.

    Args:
        identifier: The ID of the user/admin initiating the transfer.
        role: The role of the initiator ('user' or 'admin').
        source_acc: The account number from which funds are to be transferred.
        destination_acc: The account number to which funds are to be transferred.
        value: The amount of money to transfer.

    Returns:
        A string message indicating success or failure of the transaction.

    Raises:
        PermissionError: If the user is not authorized for the transaction.
        ValueError: If inputs are invalid (e.g., non-positive value, non-existent accounts, invalid role).
    """
    # 0. Validate inputs
    if not isinstance(value, (int, float)) or value <= 0:
        raise ValueError("Transfer value must be a positive number.")
    if source_acc not in ACCOUNTS_DB:
        raise ValueError(f"Source account {source_acc} not found.")
    if destination_acc not in ACCOUNTS_DB:
        raise ValueError(f"Destination account {destination_acc} not found.")
    if source_acc == destination_acc:
        raise ValueError("Source and destination accounts cannot be the same.")

    # 1. Authorization
    if role == "user":
        if ACCOUNTS_DB[source_acc]["owner_id"] != identifier:
            raise PermissionError(f"User {identifier} not authorized to transfer from account {source_acc} owned by {ACCOUNTS_DB[source_acc]['owner_id']}.")
    elif role == "admin":
        pass  # Admin is authorized
    else:
        raise ValueError(f"Invalid role: '{role}'. Must be 'user' or 'admin'.")

    # 2. Check sufficient funds in source account
    current_source_balance = ACCOUNTS_DB[source_acc]["balance"]
    if current_source_balance < value:
        return f"Transfer failed: Insufficient funds in source account {source_acc}. Has {current_source_balance:.2f}, needs {value:.2f}."

    # 3. Perform transfer
    # Attempt withdrawal from source account
    if not validate_and_update_balance(source_acc, -value):
        # This path implies an unexpected issue since funds were checked.
        # validate_and_update_balance would have printed its own error.
        return f"Transfer failed: Withdrawal of {value:.2f} from account {source_acc} was unsuccessful despite initial checks."

    # Attempt deposit into destination account
    if not validate_and_update_balance(destination_acc, value):
        # Deposit failed. Critical: Must roll back the withdrawal.
        # validate_and_update_balance for deposit would have printed its own error.
        print(f"CRITICAL: Deposit of {value:.2f} to {destination_acc} failed. Attempting to roll back withdrawal from {source_acc}.")
        
        # Attempt to roll back (add funds back to source)
        if not validate_and_update_balance(source_acc, value):
            # Rollback failed. This is a very serious state.
            error_message = (
                f"CRITICAL FAILURE: Transfer of {value:.2f} from {source_acc} to {destination_acc} failed at deposit stage. "
                f"AUTOMATIC ROLLBACK of withdrawal from {source_acc} ALSO FAILED. "
                f"Account {source_acc} balance is currently {ACCOUNTS_DB[source_acc]['balance']:.2f}. "
                f"It should have been restored to {current_source_balance:.2f}. Manual correction required."
            )
            print(error_message) # Ensure this is logged or highly visible
            return error_message
        
        return (f"Transfer failed: Deposit of {value:.2f} to account {destination_acc} was unsuccessful. "
                f"Withdrawal from {source_acc} has been successfully rolled back. "
                f"Account {source_acc} balance restored to {ACCOUNTS_DB[source_acc]['balance']:.2f}.")
    
    return (f"Transfer of {value:.2f} from {source_acc} to {destination_acc} successful. "
            f"New balances - {source_acc}: {ACCOUNTS_DB[source_acc]['balance']:.2f}, "
            f"{destination_acc}: {ACCOUNTS_DB[destination_acc]['balance']:.2f}.")
