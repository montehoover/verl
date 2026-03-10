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

def send_funds(account_holder: str, user_role: str, withdrawal_account: str, receiving_account: str, fund_amount: float) -> str:
    """
    Transfers funds between two accounts.

    Args:
        account_holder: The user ID initiating the transaction.
        user_role: The role of the user (e.g., 'user', 'admin').
        withdrawal_account: The account number to withdraw funds from.
        receiving_account: The account number to deposit funds into.
        fund_amount: The amount to transfer.

    Returns:
        A success or error message string.

    Raises:
        PermissionError: If the account_holder is not authorized.
    """
    if withdrawal_account not in ACCOUNTS_DB:
        return f"Error: Withdrawal account {withdrawal_account} not found."
    if receiving_account not in ACCOUNTS_DB:
        return f"Error: Receiving account {receiving_account} not found."

    withdrawal_acc_details = ACCOUNTS_DB[withdrawal_account]

    # Check authorization
    if not (withdrawal_acc_details["owner_id"] == account_holder or user_role == "admin"):
        raise PermissionError(f"User {account_holder} not authorized to withdraw from account {withdrawal_account}.")

    if withdrawal_acc_details["balance"] < fund_amount:
        return f"Error: Insufficient funds in account {withdrawal_account}."

    if fund_amount <= 0:
        return "Error: Fund amount must be positive."

    # Perform transaction
    ACCOUNTS_DB[withdrawal_account]["balance"] -= fund_amount
    ACCOUNTS_DB[receiving_account]["balance"] += fund_amount

    return f"Success: Transferred ${fund_amount:.2f} from {withdrawal_account} to {receiving_account}. " \
           f"New balance for {withdrawal_account}: ${ACCOUNTS_DB[withdrawal_account]['balance']:.2f}. " \
           f"New balance for {receiving_account}: ${ACCOUNTS_DB[receiving_account]['balance']:.2f}."


if __name__ == '__main__':
    print("Initial balances:")
    for acc_num, details in ACCOUNTS_DB.items():
        print(f"Account {acc_num}: Owner {details['owner_id']}, Balance ${details['balance']:.2f}")
    print("-" * 30)

    # Scenario 1: Successful transaction by owner
    try:
        result = send_funds(account_holder="USER1", user_role="user", withdrawal_account="ACC001", receiving_account="ACC002", fund_amount=100.0)
        print(f"Transaction 1: {result}")
    except PermissionError as e:
        print(f"Transaction 1 Error: {e}")
    print("-" * 30)

    # Scenario 2: Insufficient funds
    try:
        result = send_funds(account_holder="USER2", user_role="user", withdrawal_account="ACC002", receiving_account="ACC001", fund_amount=1000.0)
        print(f"Transaction 2: {result}")
    except PermissionError as e:
        print(f"Transaction 2 Error: {e}")
    print("-" * 30)

    # Scenario 3: Unauthorized transaction (wrong owner)
    try:
        result = send_funds(account_holder="USER1", user_role="user", withdrawal_account="ACC002", receiving_account="ACC001", fund_amount=50.0)
        print(f"Transaction 3: {result}")
    except PermissionError as e:
        print(f"Transaction 3 Error: {e}")
    print("-" * 30)

    # Scenario 4: Successful transaction by admin (even if not owner)
    try:
        result = send_funds(account_holder="ADMIN_USER", user_role="admin", withdrawal_account="ACC002", receiving_account="ACC001", fund_amount=50.0)
        print(f"Transaction 4: {result}")
    except PermissionError as e:
        print(f"Transaction 4 Error: {e}")
    print("-" * 30)

    # Scenario 5: Withdrawal account does not exist
    try:
        result = send_funds(account_holder="USER1", user_role="user", withdrawal_account="ACC003", receiving_account="ACC001", fund_amount=50.0)
        print(f"Transaction 5: {result}")
    except PermissionError as e:
        print(f"Transaction 5 Error: {e}")
    print("-" * 30)

    # Scenario 6: Receiving account does not exist
    try:
        result = send_funds(account_holder="USER1", user_role="user", withdrawal_account="ACC001", receiving_account="ACC003", fund_amount=50.0)
        print(f"Transaction 6: {result}")
    except PermissionError as e:
        print(f"Transaction 6 Error: {e}")
    print("-" * 30)

    # Scenario 7: Negative fund amount
    try:
        result = send_funds(account_holder="USER1", user_role="user", withdrawal_account="ACC001", receiving_account="ACC002", fund_amount=-50.0)
        print(f"Transaction 7: {result}")
    except PermissionError as e:
        print(f"Transaction 7 Error: {e}")
    print("-" * 30)


    print("Final balances:")
    for acc_num, details in ACCOUNTS_DB.items():
        print(f"Account {acc_num}: Owner {details['owner_id']}, Balance ${details['balance']:.2f}")
