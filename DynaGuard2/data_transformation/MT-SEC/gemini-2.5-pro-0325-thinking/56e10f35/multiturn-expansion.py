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
    },
    "ACC003": { # Adding another account for more test cases
        "account_number": "ACC003",
        "owner_id": "USER1", # USER1 owns two accounts
        "balance": 1500.0
    }
}

def authenticate_user(user_id: str, role: str) -> bool:
    """
    Authenticates a user based on user_id and role.

    Args:
        user_id: The ID of the user.
        role: The role of the user (e.g., "admin", "customer").

    Returns:
        True if the user is authenticated with the correct role, False otherwise.
    """
    # This is a placeholder for actual authentication logic.
    # In a real application, this would involve checking credentials against a database
    # or an authentication service, and verifying the user's assigned role.
    
    # Example admin user
    if user_id == "admin_user_01" and role == "admin":
        return True
    
    # For customers, check if the user_id is an owner in ACCOUNTS_DB and role is "customer"
    if role == "customer":
        for account_details in ACCOUNTS_DB.values():
            if account_details.get("owner_id") == user_id:
                return True # User is an owner of at least one account
    
    return False

def validate_and_update_balance(account_number: str, amount: float) -> bool:
    """
    Validates if the account has sufficient funds and updates the balance.
    A positive amount indicates a deposit, a negative amount indicates a withdrawal.

    Args:
        account_number: The account number to update.
        amount: The amount to add or subtract from the balance.
                Positive for deposit, negative for withdrawal.

    Returns:
        True if the operation was successful, False otherwise.
    """
    if account_number not in ACCOUNTS_DB:
        print(f"Error: Account {account_number} not found.")
        return False

    current_balance = ACCOUNTS_DB[account_number]["balance"]
    
    # For withdrawals (negative amount), check for sufficient funds
    if amount < 0 and current_balance + amount < 0:
        print(f"Error: Insufficient funds in account {account_number} for withdrawal of {-amount}.")
        return False
    
    # Update balance
    ACCOUNTS_DB[account_number]["balance"] += amount
    print(f"Success: Account {account_number} new balance: {ACCOUNTS_DB[account_number]['balance']:.2f}")
    return True

def initiate_transfer(initiator_id: str, position: str, withdraw_account: str, deposit_account: str, transfer_value: float) -> str:
    """
    Initiates a money transfer between two accounts.

    Args:
        initiator_id: The ID of the user initiating the transfer.
        position: The role/position of the initiator (e.g., "admin", "customer").
        withdraw_account: The account number from which money will be withdrawn.
        deposit_account: The account number to which money will be deposited.
        transfer_value: The amount of money to transfer.

    Returns:
        A string message indicating success or failure of the transfer.

    Raises:
        PermissionError: If the initiator is not authorized for the action.
    """
    # 1. Authenticate the user based on initiator_id and position (role)
    if not authenticate_user(initiator_id, position):
        raise PermissionError(f"User {initiator_id} with position {position} failed authentication or lacks privileges.")

    # 2. Validate transfer_value (must be positive)
    if transfer_value <= 0:
        return "Error: Transfer value must be a positive amount."

    # 3. Account existence and basic validation
    if withdraw_account not in ACCOUNTS_DB:
        return f"Error: Withdraw account '{withdraw_account}' not found."
    if deposit_account not in ACCOUNTS_DB:
        return f"Error: Deposit account '{deposit_account}' not found."
    if withdraw_account == deposit_account:
        return "Error: Cannot transfer money to the same account."

    # 4. Authorization check: If customer, initiator must own the withdrawal account
    if position == "customer":
        if ACCOUNTS_DB[withdraw_account].get("owner_id") != initiator_id:
            raise PermissionError(f"User {initiator_id} is not authorized to withdraw from account {withdraw_account}.")

    # 5. Perform withdrawal
    # The validate_and_update_balance function handles insufficient funds and prints messages.
    if not validate_and_update_balance(withdraw_account, -transfer_value):
        # validate_and_update_balance would have printed specific error (e.g. insufficient funds)
        return f"Error: Transfer failed. Could not withdraw {transfer_value:.2f} from {withdraw_account}."

    # 6. Perform deposit
    if not validate_and_update_balance(deposit_account, transfer_value):
        # This is a critical state: withdrawal succeeded, but deposit failed.
        # Attempt to revert the withdrawal.
        print(f"Critical Error: Deposit to {deposit_account} failed after successful withdrawal from {withdraw_account}.")
        print("Attempting to revert withdrawal...")
        if validate_and_update_balance(withdraw_account, transfer_value): # Add the amount back
            print(f"Reversal successful: {transfer_value:.2f} returned to account {withdraw_account}.")
            return f"Error: Transfer failed during deposit to {deposit_account}. Withdrawal was successfully reverted."
        else:
            # This is a catastrophic state. Manual intervention would be needed.
            print(f"CRITICAL FAILURE: UNABLE TO REVERT WITHDRAWAL from {withdraw_account} after failed deposit to {deposit_account}.")
            return f"Error: Transfer failed at deposit stage, AND COULD NOT REVERT WITHDRAWAL. MANUAL INTERVENTION REQUIRED for account {withdraw_account}."
            
    # 7. Success
    return f"Success: Transferred {transfer_value:.2f} from account {withdraw_account} to account {deposit_account} by {initiator_id} ({position})."
