ACCOUNTS_DB = {
    "ACC123": {"balance": 1000.00, "owner_id": "user001"},
    "ACC456": {"balance": 500.00, "owner_id": "user002"},
    "ACC789": {"balance": 2000.00, "owner_id": "user003"},
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
