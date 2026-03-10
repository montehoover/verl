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
