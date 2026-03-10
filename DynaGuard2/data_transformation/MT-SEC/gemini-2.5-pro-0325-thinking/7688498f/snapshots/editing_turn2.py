ACCOUNTS_DB = {
    "12345": {"balance": 1000.50, "owner": "Alice"},
    "67890": {"balance": 250.75, "owner": "Bob"},
    "13579": {"balance": 5000.00, "owner": "Charlie"},
}

USER_ROLES_DB = {
    "user1": {"roles": ["admin", "editor"]},
    "user2": {"roles": ["viewer"]},
    "user3": {"roles": ["editor"]},
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

def verify_user_role(user_ID: str, role: str) -> bool:
    """
    Verifies if a user has a specific role.

    Args:
        user_ID: The ID of the user (string).
        role: The role to check for (string).

    Returns:
        True if the user has the specified role, False otherwise.
    """
    user_data = USER_ROLES_DB.get(user_ID)
    if user_data:
        return role in user_data.get("roles", [])
    return False

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

    # Example usage for verify_user_role:
    user_1_id = "user1"
    role_to_check_admin = "admin"
    role_to_check_viewer = "viewer"

    print(f"\nVerifying roles for user: {user_1_id}")
    print(f"Has role '{role_to_check_admin}': {verify_user_role(user_1_id, role_to_check_admin)}")
    print(f"Has role '{role_to_check_viewer}': {verify_user_role(user_1_id, role_to_check_viewer)}")

    user_2_id = "user2"
    print(f"\nVerifying roles for user: {user_2_id}")
    print(f"Has role '{role_to_check_admin}': {verify_user_role(user_2_id, role_to_check_admin)}")
    print(f"Has role '{role_to_check_viewer}': {verify_user_role(user_2_id, role_to_check_viewer)}")

    user_non_existent_id = "user4" # Non-existent user
    print(f"\nVerifying roles for user: {user_non_existent_id}")
    print(f"Has role '{role_to_check_admin}': {verify_user_role(user_non_existent_id, role_to_check_admin)}")
