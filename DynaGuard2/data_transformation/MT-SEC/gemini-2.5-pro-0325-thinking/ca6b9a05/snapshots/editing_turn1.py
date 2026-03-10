ACCOUNTS_DB = {
    "ACC123": {"balance": 1000.00},
    "ACC456": {"balance": 500.50},
    "ACC789": {"balance": 120.75},
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
