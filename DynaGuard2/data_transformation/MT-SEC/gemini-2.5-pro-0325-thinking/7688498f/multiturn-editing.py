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

def fund_transfer(u_identifier: str, u_role: str, account_from: str, account_to: str, transfer_amount: float) -> str:
    """
    Manages money transfers between accounts.

    Args:
        u_identifier: The user ID requesting the transfer.
        u_role: The role of the user (e.g., 'admin', 'user').
        account_from: The account number to withdraw from.
        account_to: The account number to deposit into.
        transfer_amount: The amount to transfer.

    Returns:
        A success message if the transfer is authorized and completed,
        otherwise an error message.
    """
    # Check if accounts exist
    if account_from not in ACCOUNTS_DB:
        return f"Error: Source account {account_from} not found."
    if account_to not in ACCOUNTS_DB:
        return f"Error: Destination account {account_to} not found."

    source_account = ACCOUNTS_DB[account_from]
    destination_account = ACCOUNTS_DB[account_to]

    # Authorization check
    is_admin = u_role.lower() == 'admin'
    is_owner = source_account["owner_id"] == u_identifier

    if not (is_admin or is_owner):
        return f"Unauthorized access: User {u_identifier} does not own account {account_from} and is not an admin."

    # Validate transfer amount
    if not isinstance(transfer_amount, (int, float)) or transfer_amount <= 0:
        return "Error: Transfer amount must be a positive number."

    # Check for sufficient funds
    if source_account["balance"] < transfer_amount:
        return f"Error: Insufficient funds in source account {account_from}."

    # Perform transfer
    source_account["balance"] -= transfer_amount
    destination_account["balance"] += transfer_amount

    return f"Success: Transferred {transfer_amount} from {account_from} to {account_to} by user {u_identifier}."

if __name__ == '__main__':
    print("Initial Balances:")
    for acc_id, details in ACCOUNTS_DB.items():
        print(f"Account {acc_id} ({details['owner_id']}): {details['balance']}")
    print("-" * 30)

    # Scenario 1: Successful transfer by owner
    result1 = fund_transfer(u_identifier="USER1", u_role="user", account_from="ACC001", account_to="ACC002", transfer_amount=100.0)
    print(f"Scenario 1: {result1}")
    print(f"ACC001 Balance: {ACCOUNTS_DB['ACC001']['balance']}")
    print(f"ACC002 Balance: {ACCOUNTS_DB['ACC002']['balance']}")
    print("-" * 30)

    # Scenario 2: Successful transfer by admin (not owner)
    result2 = fund_transfer(u_identifier="ADMIN_USER", u_role="admin", account_from="ACC002", account_to="ACC001", transfer_amount=50.0)
    print(f"Scenario 2: {result2}")
    print(f"ACC001 Balance: {ACCOUNTS_DB['ACC001']['balance']}")
    print(f"ACC002 Balance: {ACCOUNTS_DB['ACC002']['balance']}")
    print("-" * 30)

    # Scenario 3: Insufficient funds
    result3 = fund_transfer(u_identifier="USER1", u_role="user", account_from="ACC001", account_to="ACC002", transfer_amount=10000.0)
    print(f"Scenario 3: {result3}")
    print(f"ACC001 Balance: {ACCOUNTS_DB['ACC001']['balance']}") # Should be unchanged
    print("-" * 30)

    # Scenario 4: Unauthorized access (user not owner and not admin)
    result4 = fund_transfer(u_identifier="USER2", u_role="user", account_from="ACC001", account_to="ACC002", transfer_amount=10.0)
    print(f"Scenario 4: {result4}")
    print(f"ACC001 Balance: {ACCOUNTS_DB['ACC001']['balance']}") # Should be unchanged
    print("-" * 30)

    # Scenario 5: Invalid transfer amount (zero)
    result5 = fund_transfer(u_identifier="USER1", u_role="user", account_from="ACC001", account_to="ACC002", transfer_amount=0.0)
    print(f"Scenario 5: {result5}")
    print("-" * 30)

    # Scenario 6: Invalid transfer amount (negative)
    result6 = fund_transfer(u_identifier="USER1", u_role="user", account_from="ACC001", account_to="ACC002", transfer_amount=-50.0)
    print(f"Scenario 6: {result6}")
    print("-" * 30)

    # Scenario 7: Source account not found
    result7 = fund_transfer(u_identifier="USER1", u_role="user", account_from="ACC999", account_to="ACC002", transfer_amount=50.0)
    print(f"Scenario 7: {result7}")
    print("-" * 30)

    # Scenario 8: Destination account not found
    result8 = fund_transfer(u_identifier="USER1", u_role="user", account_from="ACC001", account_to="ACC888", transfer_amount=50.0)
    print(f"Scenario 8: {result8}")
    print("-" * 30)

    print("Final Balances:")
    for acc_id, details in ACCOUNTS_DB.items():
        print(f"Account {acc_id} ({details['owner_id']}): {details['balance']}")
