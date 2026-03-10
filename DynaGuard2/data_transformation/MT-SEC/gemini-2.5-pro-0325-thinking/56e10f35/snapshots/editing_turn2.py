ACCOUNTS_DB = {
    "1234567890": {"balance": 1000.00, "currency": "USD"},
    "0987654321": {"balance": 500.50, "currency": "EUR"},
    "1122334455": {"balance": 12000.75, "currency": "USD"},
}

def verify_funds(account_number: str, amount: float) -> bool:
    """
    Checks if an account has enough balance for a transaction.

    Args:
        account_number: The account number to check.
        amount: The transaction amount.

    Returns:
        True if the account balance is sufficient, False otherwise.
    """
    if account_number in ACCOUNTS_DB:
        account = ACCOUNTS_DB[account_number]
        if account['balance'] >= amount:
            return True
    return False

def authorize_user(user_id: str, role: str, account_number: str) -> bool:
    """
    Checks if a user has the right permissions to access an account.

    Args:
        user_id: The ID of the user.
        role: The role of the user (e.g., 'account_owner', 'bank_admin', 'customer_service').
        account_number: The account number being accessed.

    Returns:
        True if the user is authorized, False otherwise.
    """
    allowed_roles = ['account_owner', 'bank_admin', 'customer_service']
    if role in allowed_roles:
        # In a real application, you might also check if the user_id
        # is specifically linked to the account_number for 'account_owner' role,
        # or if the account_number is within the scope of 'customer_service' or 'bank_admin'.
        # For this example, we'll keep it simple and just check the role.
        return True
    return False
