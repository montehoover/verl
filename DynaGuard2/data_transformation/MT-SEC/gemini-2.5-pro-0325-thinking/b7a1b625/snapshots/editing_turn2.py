ACCOUNTS_DB = {
    "1234567890": {"balance": 1000, "owner": "Alice"},
    "0987654321": {"balance": 500, "owner": "Bob"},
    "1122334455": {"balance": 2500, "owner": "Charlie"},
}

def account_exists(account_number: str) -> bool:
    """
    Checks if a given account number exists in the system.

    Args:
        account_number: The account number to check (string).

    Returns:
        True if the account exists in ACCOUNTS_DB, False otherwise.
    """
    return account_number in ACCOUNTS_DB

def get_balance(account_number: str) -> int | None:
    """
    Retrieves the balance of a given account number.

    Args:
        account_number: The account number to retrieve the balance for (string).

    Returns:
        The balance of the account as an integer if the account exists, 
        otherwise None.
    """
    if account_exists(account_number):
        return ACCOUNTS_DB[account_number]["balance"]
    return None
