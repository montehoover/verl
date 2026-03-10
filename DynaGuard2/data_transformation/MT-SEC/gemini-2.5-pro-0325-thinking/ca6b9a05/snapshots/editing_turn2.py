ACCOUNTS_DB = {
    "ACC123": {"balance": 1000.00, "owner_id": "user001"},
    "ACC456": {"balance": 500.50, "owner_id": "user002"},
    "ACC789": {"balance": 120.75, "owner_id": "user003"},
}

def check_account_balance(account_number: str, amount: float) -> bool:
    """
    Checks if an account has sufficient funds for a transaction.

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
    Verifies if a user has the right permissions for an account.

    Args:
        user_id: The ID of the user.
        role: The role of the user (e.g., 'account_owner', 'bank_admin', 'customer_service').
        account_number: The account number to check permissions for.

    Returns:
        True if the user has necessary permissions, False otherwise.
    """
    if account_number not in ACCOUNTS_DB:
        return False  # Account does not exist

    account_details = ACCOUNTS_DB[account_number]

    if role == "bank_admin":
        return True  # Bank admins can access any account
    
    if role == "customer_service":
        # Customer service can also access any account for support purposes (example rule)
        return True

    if role == "account_owner":
        return account_details.get("owner_id") == user_id

    return False # Default to no access if role is unknown or doesn't grant access
