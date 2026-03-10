# Placeholder for account data. In a real application, this would be a database.
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

def execute_transfer(employee_id: str, role_value: str, source_acct: str, target_acct: str, total_amount: float) -> str:
    """
    Facilitates money transfers between accounts.

    Args:
        employee_id: The ID of the employee performing the transfer.
        role_value: The role of the employee, used for authorization.
        source_acct: The account number from which money is transferred.
        target_acct: The account number to which money is transferred.
        total_amount: The amount of money to transfer.

    Returns:
        A success message if the transfer is successful.

    Raises:
        PermissionError: If the employee role is not authorized.
        ValueError: If the amount is negative or zero.
    """
    # Authorization check (placeholder logic)
    # In a real system, this would involve more robust role/permission checking.
    if role_value not in ["employee", "manager", "admin"]: # Example authorized roles
        raise PermissionError(f"Employee {employee_id} with role {role_value} is not authorized to perform transfers.")

    if total_amount <= 0:
        return "Error: Transfer amount must be positive."

    # Validate accounts
    if source_acct not in ACCOUNTS_DB:
        return f"Error: Source account {source_acct} does not exist."
    if target_acct not in ACCOUNTS_DB:
        return f"Error: Target account {target_acct} does not exist."
    
    if source_acct == target_acct:
        return "Error: Source and target accounts cannot be the same."

    # Check for sufficient funds
    if ACCOUNTS_DB[source_acct]["balance"] < total_amount:
        return f"Error: Insufficient funds in source account {source_acct}."

    # Perform transfer
    ACCOUNTS_DB[source_acct]["balance"] -= total_amount
    ACCOUNTS_DB[target_acct]["balance"] += total_amount

    return f"Success: Transferred {total_amount} from {source_acct} to {target_acct}. Employee: {employee_id}."
