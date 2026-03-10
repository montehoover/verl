# Sample accounts database for demonstration
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
        user_id (str): The user's unique identifier
        password (str): The user's password
        
    Returns:
        bool: True if authentication successful, False otherwise
    """
    # This is a placeholder implementation
    # In production, this would check against a secure database
    # with properly hashed passwords
    
    # Example hardcoded users for demonstration
    # Never store passwords in plain text in production!
    valid_users = {
        "user123": "securepass456",
        "admin": "adminpass789",
        "john_doe": "password123"
    }
    
    # Check if user exists and password matches
    if user_id in valid_users and valid_users[user_id] == password:
        return True
    
    return False


def validate_and_update_balance(account_number: str, amount: float) -> bool:
    """
    Validates and updates the account balance for a given account.
    
    Args:
        account_number (str): The account number to update
        amount (float): The amount to deduct from the account (positive for debit, negative for credit)
        
    Returns:
        bool: True if the update was successful, False otherwise
    """
    # Check if account exists
    if account_number not in ACCOUNTS_DB:
        return False
    
    # Get current balance
    current_balance = ACCOUNTS_DB[account_number]["balance"]
    
    # For debits (positive amounts), check if sufficient funds
    if amount > 0 and current_balance < amount:
        return False
    
    # Update the balance
    ACCOUNTS_DB[account_number]["balance"] = current_balance - amount
    
    return True


def execute_transfer(employee_id: str, role_value: str, source_acct: str, target_acct: str, total_amount: float) -> str:
    """
    Executes a money transfer between accounts.
    
    Args:
        employee_id (str): The employee's ID executing the transfer
        role_value (str): The role/permission level of the employee
        source_acct (str): The source account number
        target_acct (str): The target account number
        total_amount (float): The amount to transfer
        
    Returns:
        str: Success message upon successful transfer
        
    Raises:
        PermissionError: If the employee lacks authorization
    """
    # Define authorized roles
    authorized_roles = ["manager", "admin", "teller", "supervisor"]
    
    # Check if employee has permission
    if role_value.lower() not in authorized_roles:
        raise PermissionError(f"Employee {employee_id} with role {role_value} is not authorized to perform transfers")
    
    # Verify source and target accounts exist
    if source_acct not in ACCOUNTS_DB:
        raise ValueError(f"Source account {source_acct} does not exist")
    
    if target_acct not in ACCOUNTS_DB:
        raise ValueError(f"Target account {target_acct} does not exist")
    
    # Check if source account has sufficient balance
    if ACCOUNTS_DB[source_acct]["balance"] < total_amount:
        raise ValueError(f"Insufficient funds in source account {source_acct}")
    
    # Perform the transfer
    ACCOUNTS_DB[source_acct]["balance"] -= total_amount
    ACCOUNTS_DB[target_acct]["balance"] += total_amount
    
    return f"Transfer successful: ${total_amount:.2f} transferred from {source_acct} to {target_acct}"
