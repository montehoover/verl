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

def has_sufficient_funds(account_number: str, amount: float) -> bool:
    """
    Checks if an account has enough balance for a transaction.

    Args:
        account_number: The account number to check.
        amount: The transaction amount.

    Returns:
        True if the account balance is greater than or equal to the amount, 
        False otherwise.
    """
    if account_number in ACCOUNTS_DB:
        account = ACCOUNTS_DB[account_number]
        return account.get("balance", 0.0) >= amount
    return False

def verify_user_role(user_id: str, role: str, account_number: str) -> bool:
    """
    Verifies if a user has the right permissions to access an account.

    Args:
        user_id: The ID of the user.
        role: The role of the user (e.g., 'account_owner', 'bank_admin', 'customer_service').
        account_number: The account number to check access for.

    Returns:
        True if the user has necessary permissions, False otherwise.
    """
    if account_number not in ACCOUNTS_DB:
        return False

    account_details = ACCOUNTS_DB[account_number]

    if role == "bank_admin":
        return True
    elif role == "customer_service":
        # Customer service might have broad access for support, or specific limited access.
        # For this example, we'll grant broad access similar to admin.
        return True
    elif role == "account_owner":
        return account_details.get("owner_id") == user_id
    
    return False

def process_transfer(uid: str, role: str, source_account: str, destination_account: str, transfer_amount: float) -> str:
    """
    Processes a money transfer between two accounts.

    Args:
        uid: The user ID initiating the transfer.
        role: The user's role.
        source_account: The account number to withdraw from.
        destination_account: The account number to deposit into.
        transfer_amount: The amount to transfer.

    Returns:
        A success message if the transfer is authorized and completed,
        otherwise an error message.
    """
    # Verify user authorization for the source account
    if not verify_user_role(uid, role, source_account):
        return "Error: User not authorized to access source account."

    # Check if source and destination accounts exist
    if source_account not in ACCOUNTS_DB:
        return f"Error: Source account {source_account} not found."
    if destination_account not in ACCOUNTS_DB:
        return f"Error: Destination account {destination_account} not found."

    # Check for sufficient funds
    if not has_sufficient_funds(source_account, transfer_amount):
        return f"Error: Insufficient funds in source account {source_account}."

    # Perform the transfer
    ACCOUNTS_DB[source_account]["balance"] -= transfer_amount
    ACCOUNTS_DB[destination_account]["balance"] += transfer_amount

    return f"Success: Transferred {transfer_amount} from {source_account} to {destination_account}."
