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

def verify_user_role(user_id: str, role: str, account_number: str) -> bool:
    """
    Verifies if a user has the right permissions for an account.
    (This is a simplified version for the move_funds function context)
    """
    if account_number not in ACCOUNTS_DB:
        return False  # Account does not exist

    account_details = ACCOUNTS_DB[account_number]

    if role == "bank_admin":
        return True
    
    if role == "account_owner":
        return account_details.get("owner_id") == user_id

    return False

def check_account_balance(account_number: str, amount: float) -> bool:
    """
    Checks if an account has sufficient funds for a transaction.
    """
    if account_number in ACCOUNTS_DB:
        account_balance = ACCOUNTS_DB[account_number].get("balance", 0.0)
        return account_balance >= amount
    return False

def move_funds(u_id: str, u_role: str, source_account: str, destination_account: str, money: float) -> str:
    """
    Handles transferring money between accounts.

    Args:
        u_id: The identifier of the individual requesting the transfer.
        u_role: The role held by the individual.
        source_account: The account from which the funds will be removed.
        destination_account: The account where the funds will be deposited.
        money: The monetary value to transfer.

    Returns:
        A success message if the transfer is authorized and completed,
        otherwise an error message.
    """
    # Verify user authorization for the source account
    if not verify_user_role(u_id, u_role, source_account):
        return "Error: User not authorized to access source account."

    # Check if source and destination accounts exist
    if source_account not in ACCOUNTS_DB:
        return f"Error: Source account {source_account} not found."
    if destination_account not in ACCOUNTS_DB:
        return f"Error: Destination account {destination_account} not found."

    # Check for non-positive transfer amount
    if money <= 0:
        return "Error: Transfer amount must be positive."

    # Check for sufficient funds in the source account
    if not check_account_balance(source_account, money):
        return f"Error: Insufficient funds in source account {source_account}."

    # Perform the transfer
    ACCOUNTS_DB[source_account]["balance"] -= money
    ACCOUNTS_DB[destination_account]["balance"] += money

    return f"Success: Transferred {money} from {source_account} to {destination_account}."
