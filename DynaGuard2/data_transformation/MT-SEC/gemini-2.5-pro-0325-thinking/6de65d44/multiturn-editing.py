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

# Define roles
ROLE_ACCOUNT_OWNER = "account_owner"
ROLE_BANK_ADMIN = "bank_admin"
ROLE_CUSTOMER_SERVICE = "customer_service"

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
        account_balance = ACCOUNTS_DB[account_number].get("balance", 0.0)
        return account_balance >= amount
    return False

def verify_user_role(user_id: str, role: str, account_number: str) -> bool:
    """
    Verifies if a user has the right permissions to access or modify an account.

    Args:
        user_id: The ID of the user.
        role: The role of the user (e.g., 'account_owner', 'bank_admin', 'customer_service').
        account_number: The account number to check permissions for.

    Returns:
        True if the user has the necessary permissions, False otherwise.
    """
    if account_number not in ACCOUNTS_DB:
        return False  # Account does not exist

    account_info = ACCOUNTS_DB[account_number]

    if role == ROLE_BANK_ADMIN:
        return True  # Bank admins can access any account

    if role == ROLE_CUSTOMER_SERVICE:
        # Customer service might have specific rules, for now, let's assume they can access.
        # This could be expanded, e.g., read-only access or specific account types.
        return True

    if role == ROLE_ACCOUNT_OWNER:
        return account_info.get("owner_id") == user_id

    return False  # Unknown role or insufficient permissions

def move_money(person_id: str, role: str, source_account: str, destination_account: str, transfer_amount: float) -> str:
    """
    Handles transferring funds between accounts.

    Args:
        person_id: The user ID initiating the transaction.
        role: The user role.
        source_account: The account to withdraw funds from.
        destination_account: The account to deposit the funds into.
        transfer_amount: The amount to transfer.

    Returns:
        A success message if the transfer is confirmed, otherwise an error message.
    """
    # Verify user's permission to access the source account
    if not verify_user_role(person_id, role, source_account):
        return "Error: Unauthorized access to source account."

    # Check if the source account exists (already covered by verify_user_role, but good for clarity)
    if source_account not in ACCOUNTS_DB:
        return f"Error: Source account {source_account} not found."

    # Check if the destination account exists
    if destination_account not in ACCOUNTS_DB:
        return f"Error: Destination account {destination_account} not found."

    # Ensure the transfer amount is positive
    if transfer_amount <= 0:
        return "Error: Transfer amount must be positive."

    # Check for sufficient funds in the source account
    if not has_sufficient_funds(source_account, transfer_amount):
        return f"Error: Insufficient funds in source account {source_account}."

    # Perform the transfer
    ACCOUNTS_DB[source_account]["balance"] -= transfer_amount
    ACCOUNTS_DB[destination_account]["balance"] += transfer_amount

    return f"Success: Transferred {transfer_amount:.2f} from {source_account} to {destination_account}."
