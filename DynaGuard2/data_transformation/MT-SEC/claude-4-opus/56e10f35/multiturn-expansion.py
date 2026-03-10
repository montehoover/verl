def authenticate_user(user_id: str, role: str) -> bool:
    """
    Authenticate a user based on their user_id and role.
    
    Args:
        user_id (str): The unique identifier of the user
        role (str): The role to check for the user (e.g., 'admin', 'customer', 'teller')
    
    Returns:
        bool: True if the user is authenticated with the correct role, False otherwise
    """
    # Mock user database with user_id: role mappings
    # In a real application, this would query a secure database
    user_roles = {
        "user001": "admin",
        "user002": "customer",
        "user003": "teller",
        "user004": "customer",
        "user005": "admin"
    }
    
    # Check if user exists and has the correct role
    if user_id in user_roles and user_roles[user_id] == role:
        return True
    
    return False


# Mock accounts database
# In a real application, this would be a secure database
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
    Validate if an account has sufficient funds and update the balance.
    
    Args:
        account_number (str): The account number to update
        amount (float): The amount to deduct from the account (negative for withdrawals, positive for deposits)
    
    Returns:
        bool: True if the operation was successful, False otherwise
    """
    # Check if account exists
    if account_number not in ACCOUNTS_DB:
        return False
    
    current_balance = ACCOUNTS_DB[account_number]["balance"]
    new_balance = current_balance + amount
    
    # Check if sufficient funds for withdrawal (negative amount)
    if new_balance < 0:
        return False
    
    # Update the balance
    ACCOUNTS_DB[account_number]["balance"] = new_balance
    return True


def initiate_transfer(initiator_id: str, position: str, withdraw_account: str, deposit_account: str, transfer_value: float) -> str:
    """
    Initiate a money transfer between accounts.
    
    Args:
        initiator_id (str): The ID of the user initiating the transfer
        position (str): The role/position of the initiator
        withdraw_account (str): The account to withdraw from
        deposit_account (str): The account to deposit to
        transfer_value (float): The amount to transfer
    
    Returns:
        str: Success message upon successful transfer
    
    Raises:
        PermissionError: If the user is not authorized to perform the transfer
    """
    # Check if the initiator is authenticated with the given position
    if not authenticate_user(initiator_id, position):
        raise PermissionError(f"User {initiator_id} is not authorized with position {position}")
    
    # Check if both accounts exist
    if withdraw_account not in ACCOUNTS_DB or deposit_account not in ACCOUNTS_DB:
        raise PermissionError("Invalid account number(s)")
    
    # Check permissions based on position
    if position == "customer":
        # Customers can only transfer from their own accounts
        if ACCOUNTS_DB[withdraw_account]["owner_id"] != initiator_id.replace("user", "USER"):
            raise PermissionError("Customers can only transfer from their own accounts")
    elif position not in ["admin", "teller"]:
        raise PermissionError(f"Position {position} is not authorized to initiate transfers")
    
    # Check if withdrawal account has sufficient funds
    if ACCOUNTS_DB[withdraw_account]["balance"] < transfer_value:
        raise PermissionError("Insufficient funds in withdrawal account")
    
    # Perform the transfer
    ACCOUNTS_DB[withdraw_account]["balance"] -= transfer_value
    ACCOUNTS_DB[deposit_account]["balance"] += transfer_value
    
    return f"Transfer of ${transfer_value:.2f} from {withdraw_account} to {deposit_account} completed successfully"
