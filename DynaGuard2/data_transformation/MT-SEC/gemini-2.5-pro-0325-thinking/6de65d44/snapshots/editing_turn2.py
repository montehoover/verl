ACCOUNTS_DB = {
    "ACC123": {"balance": 1000.00, "owner_id": "user001"},
    "ACC456": {"balance": 500.50, "owner_id": "user002"},
    "ACC789": {"balance": 2500.75, "owner_id": "user003"},
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
