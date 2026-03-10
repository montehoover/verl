# Placeholder for a database of accounts and their balances
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
    if user_id == "admin_user" and role == "admin":
        return True
    if user_id == "customer123" and role == "customer":
        return True
    return False

def validate_and_update_balance(account_number: str, amount: float) -> bool:
    """
    Validates if an account has sufficient funds and updates the balance.

    Args:
        account_number: The account number to update.
        amount: The amount to add (positive for deposit, negative for withdrawal).

    Returns:
        True if the operation was successful, False otherwise.
    """
    if account_number not in ACCOUNTS_DB:
        print(f"Error: Account {account_number} not found.")
        return False

    current_balance = ACCOUNTS_DB[account_number]["balance"]
    
    if current_balance + amount < 0:
        print(f"Error: Insufficient funds in account {account_number} for transaction of {amount}.")
        return False
    
    ACCOUNTS_DB[account_number]["balance"] += amount
    print(f"Success: Account {account_number} updated. New balance: {ACCOUNTS_DB[account_number]['balance']:.2f}")
    return True

def shift_funds(initiator_id: str, initiator_role: str, withdraw_account: str, deposit_account: str, transfer_value: float) -> str:
    """
    Transfers funds between two accounts after validating user permissions and balances.

    Args:
        initiator_id: The ID of the user initiating the transfer.
        initiator_role: The role of the user initiating the transfer.
        withdraw_account: The account number from which funds will be withdrawn.
        deposit_account: The account number to which funds will be deposited.
        transfer_value: The amount of money to transfer.

    Returns:
        A success message if the transfer is successful.

    Raises:
        PermissionError: If the initiator is not authorized to perform the transfer.
        ValueError: If accounts are invalid or funds are insufficient.
    """
    if not authenticate_user(initiator_id, initiator_role):
        raise PermissionError(f"User {initiator_id} with role {initiator_role} failed authentication.")

    if withdraw_account not in ACCOUNTS_DB or deposit_account not in ACCOUNTS_DB:
        raise ValueError("Invalid account number provided for withdrawal or deposit.")

    # Check ownership or admin rights for withdrawal
    if ACCOUNTS_DB[withdraw_account]["owner_id"] != initiator_id and initiator_role != "admin":
        raise PermissionError(f"User {initiator_id} is not authorized to withdraw from account {withdraw_account}.")

    if transfer_value <= 0:
        raise ValueError("Transfer value must be positive.")

    # Perform withdrawal
    if not validate_and_update_balance(withdraw_account, -transfer_value):
        # validate_and_update_balance prints its own error, but we can raise a more specific one here
        raise ValueError(f"Failed to withdraw {transfer_value} from {withdraw_account}. Insufficient funds or other error.")

    # Perform deposit
    if not validate_and_update_balance(deposit_account, transfer_value):
        # If deposit fails, we should ideally roll back the withdrawal.
        # For simplicity in this example, we'll just raise an error.
        # In a real system, this would be a transactional operation.
        ACCOUNTS_DB[withdraw_account]["balance"] += transfer_value # Rollback withdrawal
        raise ValueError(f"Failed to deposit {transfer_value} to {deposit_account}. Rolling back withdrawal.")

    return f"Successfully transferred {transfer_value:.2f} from {withdraw_account} to {deposit_account} by {initiator_id}."
