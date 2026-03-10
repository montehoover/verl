ACCOUNTS_DB = {
    "ACC001": {
        "account_number": "ACC001",
        "owner_id": "USER1",
        "balance": 1000.0
    },
    "ACC002": {
        "account_number": "ACC002",
        "owner_id": "USER2",
        "balance": 500.0
    }
}


def _is_authorized(u_id: str, u_role: str, source_account_id: str, source_account_owner_id: str) -> bool:
    """
    Checks if the user is authorized to perform an action on the source account.

    Args:
        u_id: The identifier of the individual requesting the action.
        u_role: The role held by the individual.
        source_account_id: The identifier of the source account.
        source_account_owner_id: The owner_id of the source account.

    Returns:
        True if authorized, False otherwise.

    Raises:
        PermissionError: If an 'account_owner' tries to access an account they do not own.
    """
    if u_role == 'bank_admin':
        return True
    if u_role == 'account_owner':
        if source_account_owner_id == u_id:
            return True
        else:
            # Account owner trying to move funds from an account they don't own
            raise PermissionError(f"Error: User {u_id} is not authorized to access account {source_account_id}.")
    # For other roles like 'customer_service' or any undefined role
    return False


def _update_account_balances(source_account_data: dict, destination_account_data: dict, amount: float):
    """
    Updates the balances in the provided account data dictionaries.
    Modifies the dictionaries in-place.

    Args:
        source_account_data: Dictionary for the source account.
        destination_account_data: Dictionary for the destination account.
        amount: The monetary value to transfer.
    """
    source_account_data["balance"] -= amount
    destination_account_data["balance"] += amount


def move_funds(u_id: str, u_role: str, source_account: str, destination_account: str, money: float):
    """
    Manages transferring money between accounts in a financial system.

    Args:
        u_id: The identifier of the individual requesting the transfer.
        u_role: The role held by the individual (e.g., 'account_owner', 'bank_admin', 'customer_service').
        source_account: Indicates the account from which the funds will be removed.
        destination_account: Specifies the account where the funds will be deposited.
        money: Represents the monetary value to transfer.

    Returns:
        A success message if the transfer is authorized and completed.
        An error message if authorization is successful but other conditions (e.g., insufficient funds, invalid account) fail.

    Raises:
        PermissionError: For an unauthorized access attempt.
        ValueError: If the amount to transfer is negative.
    """
    if source_account not in ACCOUNTS_DB or destination_account not in ACCOUNTS_DB:
        return "Error: Invalid source or destination account."

    source_acc_data = ACCOUNTS_DB[source_account]

    # Authorization checks
    try:
        # _is_authorized will raise PermissionError if an account owner tries to access an unowned account,
        # or return False for other unauthorized roles (e.g. 'customer_service').
        if not _is_authorized(u_id, u_role, source_account, source_acc_data["owner_id"]):
            raise PermissionError(f"Error: Role {u_role} is not authorized to perform this transfer.")
    except PermissionError: # Re-raise the specific PermissionError from _is_authorized or the one from above
        raise

    if money < 0:
        return "Error: Transfer amount cannot be negative."

    if source_acc_data["balance"] < money: # Use pre-fetched source_acc_data
        return f"Error: Insufficient funds in source account {source_account}."

    destination_acc_data = ACCOUNTS_DB[destination_account]
    # Perform the transfer
    _update_account_balances(source_acc_data, destination_acc_data, money)

    return f"Success: Transferred {money} from {source_account} to {destination_account}."

if __name__ == '__main__':
    # Example Usage (for testing purposes)
    print("Initial ACCOUNTS_DB state:")
    for acc_id, details in ACCOUNTS_DB.items():
        print(f"  {acc_id}: {details}")
    print("-" * 30)

    # Test case 1: Successful transfer by account owner
    print("Test Case 1: Account owner successful transfer")
    try:
        result = move_funds(u_id="USER1", u_role="account_owner", source_account="ACC001", destination_account="ACC002", money=100.0)
        print(f"Result: {result}")
    except PermissionError as e:
        print(f"Error: {e}")
    print(f"ACC001 Balance: {ACCOUNTS_DB['ACC001']['balance']}")
    print(f"ACC002 Balance: {ACCOUNTS_DB['ACC002']['balance']}")
    print("-" * 30)

    # Test case 2: Successful transfer by bank admin
    print("Test Case 2: Bank admin successful transfer")
    try:
        result = move_funds(u_id="ADMIN007", u_role="bank_admin", source_account="ACC002", destination_account="ACC001", money=50.0)
        print(f"Result: {result}")
    except PermissionError as e:
        print(f"Error: {e}")
    print(f"ACC001 Balance: {ACCOUNTS_DB['ACC001']['balance']}")
    print(f"ACC002 Balance: {ACCOUNTS_DB['ACC002']['balance']}")
    print("-" * 30)

    # Test case 3: Insufficient funds
    print("Test Case 3: Insufficient funds")
    try:
        result = move_funds(u_id="USER1", u_role="account_owner", source_account="ACC001", destination_account="ACC002", money=10000.0)
        print(f"Result: {result}")
    except PermissionError as e:
        print(f"Error: {e}")
    print(f"ACC001 Balance: {ACCOUNTS_DB['ACC001']['balance']}") # Should be unchanged
    print("-" * 30)

    # Test case 4: Unauthorized access by role
    print("Test Case 4: Unauthorized access by role (customer_service)")
    try:
        result = move_funds(u_id="CS001", u_role="customer_service", source_account="ACC001", destination_account="ACC002", money=10.0)
        print(f"Result: {result}")
    except PermissionError as e:
        print(f"Error: {e}")
    print(f"ACC001 Balance: {ACCOUNTS_DB['ACC001']['balance']}") # Should be unchanged
    print("-" * 30)

    # Test case 5: Unauthorized access by account owner (wrong account)
    print("Test Case 5: Unauthorized access by account owner (wrong account)")
    try:
        result = move_funds(u_id="USER2", u_role="account_owner", source_account="ACC001", destination_account="ACC002", money=10.0)
        print(f"Result: {result}")
    except PermissionError as e:
        print(f"Error: {e}")
    print(f"ACC001 Balance: {ACCOUNTS_DB['ACC001']['balance']}") # Should be unchanged
    print("-" * 30)

    # Test case 6: Invalid source account
    print("Test Case 6: Invalid source account")
    try:
        result = move_funds(u_id="USER1", u_role="account_owner", source_account="ACC999", destination_account="ACC002", money=10.0)
        print(f"Result: {result}")
    except PermissionError as e:
        print(f"Error: {e}")
    print("-" * 30)

    # Test case 7: Invalid destination account
    print("Test Case 7: Invalid destination account")
    try:
        result = move_funds(u_id="USER1", u_role="account_owner", source_account="ACC001", destination_account="ACC888", money=10.0)
        print(f"Result: {result}")
    except PermissionError as e:
        print(f"Error: {e}")
    print("-" * 30)
    
    # Test case 8: Negative money transfer
    print("Test Case 8: Negative money transfer")
    try:
        result = move_funds(u_id="USER1", u_role="account_owner", source_account="ACC001", destination_account="ACC002", money=-50.0)
        print(f"Result: {result}")
    except PermissionError as e:
        print(f"Error: {e}")
    print(f"ACC001 Balance: {ACCOUNTS_DB['ACC001']['balance']}") # Should be unchanged
    print(f"ACC002 Balance: {ACCOUNTS_DB['ACC002']['balance']}") # Should be unchanged
    print("-" * 30)

    print("Final ACCOUNTS_DB state:")
    for acc_id, details in ACCOUNTS_DB.items():
        print(f"  {acc_id}: {details}")
