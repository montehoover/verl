ACCOUNTS_DB = {
    "1234567890": {"owner_id": "user1", "balance": 1000.00},
    "0987654321": {"owner_id": "user2", "balance": 500.50},
    "1122334455": {"owner_id": "user3", "balance": 12345.67},
}

def get_account_details(account_number: str) -> dict:
    """
    Retrieves account details for a given account number.

    Args:
        account_number: The account number to look up.

    Returns:
        A dictionary containing account details (owner_id, balance)
        if found, otherwise None.
    """
    return ACCOUNTS_DB.get(account_number)

def check_funds(account_number: str, amount: float) -> bool:
    """
    Checks if an account has sufficient funds for a transaction.

    Args:
        account_number: The account number to check.
        amount: The transaction amount.

    Returns:
        True if the balance is sufficient, False otherwise.
    """
    account_details = get_account_details(account_number)
    if account_details:
        return account_details["balance"] >= amount
    return False

if __name__ == '__main__':
    # Example usage:
    account_info = get_account_details("1234567890")
    if account_info:
        print(f"Account Details: Owner ID - {account_info['owner_id']}, Balance - ${account_info['balance']:.2f}")
    else:
        print("Account not found.")

    account_info_non_existent = get_account_details("0000000000")
    if account_info_non_existent:
        print(f"Account Details: Owner ID - {account_info_non_existent['owner_id']}, Balance - ${account_info_non_existent['balance']:.2f}")
    else:
        print("Account not found.")

    # Example usage for check_funds:
    print(f"Checking funds for account 1234567890, amount 500.00: {check_funds('1234567890', 500.00)}")
    print(f"Checking funds for account 1234567890, amount 1500.00: {check_funds('1234567890', 1500.00)}")
    print(f"Checking funds for non-existent account 0000000000, amount 100.00: {check_funds('0000000000', 100.00)}")
