ACCOUNTS_DB = {
    "ACC123": {"balance": 1000.00},
    "ACC456": {"balance": 500.00},
    "ACC789": {"balance": 2000.00},
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
