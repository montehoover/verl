ACCOUNTS_DB = {
    "12345": {"balance": 1000.50, "owner": "Alice"},
    "67890": {"balance": 250.75, "owner": "Bob"},
    "13579": {"balance": 5000.00, "owner": "Charlie"},
}

def get_balance(account_number: str) -> float:
    """
    Retrieves the current balance of a given account.

    Args:
        account_number: The account number (string) to retrieve the balance for.

    Returns:
        The balance of the account as a float.
        Returns 0.0 if the account is not found.
    """
    account_info = ACCOUNTS_DB.get(account_number)
    if account_info:
        return float(account_info.get("balance", 0.0))
    return 0.0

if __name__ == '__main__':
    # Example usage:
    test_account_1 = "12345"
    balance_1 = get_balance(test_account_1)
    print(f"Balance for account {test_account_1}: {balance_1}")

    test_account_2 = "67890"
    balance_2 = get_balance(test_account_2)
    print(f"Balance for account {test_account_2}: {balance_2}")

    test_account_3 = "00000" # Non-existent account
    balance_3 = get_balance(test_account_3)
    print(f"Balance for account {test_account_3}: {balance_3}")
