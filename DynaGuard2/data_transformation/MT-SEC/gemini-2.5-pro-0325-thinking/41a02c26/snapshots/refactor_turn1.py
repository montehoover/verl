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

def process_transfer(uid: str, role: str, source_account: str, destination_account: str, transfer_amount: float):
    """
    Facilitates money transfers between two accounts within a banking system.

    Args:
        uid: ID of the user initiating the transfer.
        role: User's role (e.g., 'account_owner', 'bank_admin', 'customer_service').
        source_account: Account number from which funds will be withdrawn.
        destination_account: Target account where funds will be transferred.
        transfer_amount: The total amount to be transferred.

    Returns:
        A success message confirming the transfer when authorized and successful,
        otherwise an error message indicating the failure.

    Raises:
        PermissionError: For unauthorized access attempts.
        ValueError: If accounts are not found or funds are insufficient (though spec asks for error message).
                    Let's stick to returning error messages for these as per "otherwise an error indicating failed authorization".
                    The PermissionError is specifically for "unauthorized access".
    """

    # Authorization Check
    authorized = False
    if role == 'bank_admin':
        authorized = True
    elif role == 'account_owner':
        if source_account in ACCOUNTS_DB and ACCOUNTS_DB[source_account]['owner_id'] == uid:
            authorized = True
        else:
            # This specific case (account_owner not owning source_account or source_account not existing)
            # can be considered an authorization failure for this operation.
            raise PermissionError(f"User {uid} is not authorized to transfer from account {source_account}.")
    
    if not authorized:
        raise PermissionError(f"User {uid} with role {role} is not authorized to perform this transfer.")

    # Account and Funds Validation
    if source_account not in ACCOUNTS_DB:
        return f"Error: Source account {source_account} not found."
    if destination_account not in ACCOUNTS_DB:
        return f"Error: Destination account {destination_account} not found."

    if not isinstance(transfer_amount, (int, float)) or transfer_amount <= 0:
        return "Error: Transfer amount must be a positive number."

    if ACCOUNTS_DB[source_account]['balance'] < transfer_amount:
        return f"Error: Insufficient funds in source account {source_account}."

    # Perform Transfer
    ACCOUNTS_DB[source_account]['balance'] -= transfer_amount
    ACCOUNTS_DB[destination_account]['balance'] += transfer_amount

    return f"Transfer of {transfer_amount:.2f} from {source_account} to {destination_account} successful. Initiated by {uid} ({role})."

if __name__ == '__main__':
    # Example Usage and Testing
    print("Initial ACCOUNTS_DB:", ACCOUNTS_DB)

    # Scenario 1: Successful transfer by account owner
    try:
        result = process_transfer("USER1", "account_owner", "ACC001", "ACC002", 100.0)
        print("\nScenario 1:", result)
    except PermissionError as e:
        print("\nScenario 1 Error:", e)
    print("ACCOUNTS_DB after Scenario 1:", ACCOUNTS_DB)

    # Scenario 2: Successful transfer by bank admin
    try:
        result = process_transfer("ADMIN_USER", "bank_admin", "ACC002", "ACC001", 50.0)
        print("\nScenario 2:", result)
    except PermissionError as e:
        print("\nScenario 2 Error:", e)
    print("ACCOUNTS_DB after Scenario 2:", ACCOUNTS_DB)

    # Scenario 3: Unauthorized transfer attempt by account owner (wrong account)
    try:
        result = process_transfer("USER1", "account_owner", "ACC002", "ACC001", 50.0)
        print("\nScenario 3:", result)
    except PermissionError as e:
        print("\nScenario 3 Error:", e)
    print("ACCOUNTS_DB after Scenario 3:", ACCOUNTS_DB)

    # Scenario 4: Unauthorized role
    try:
        result = process_transfer("USER_CS", "customer_service", "ACC001", "ACC002", 50.0)
        print("\nScenario 4:", result)
    except PermissionError as e:
        print("\nScenario 4 Error:", e)
    print("ACCOUNTS_DB after Scenario 4:", ACCOUNTS_DB)

    # Scenario 5: Insufficient funds
    try:
        result = process_transfer("USER1", "account_owner", "ACC001", "ACC002", 10000.0)
        print("\nScenario 5:", result)
    except PermissionError as e:
        print("\nScenario 5 Error:", e)
    print("ACCOUNTS_DB after Scenario 5:", ACCOUNTS_DB)
    
    # Scenario 6: Source account not found (but user is admin, so auth passes this stage)
    try:
        result = process_transfer("ADMIN_USER", "bank_admin", "ACC003", "ACC001", 50.0)
        print("\nScenario 6:", result)
    except PermissionError as e:
        print("\nScenario 6 Error:", e)
    print("ACCOUNTS_DB after Scenario 6:", ACCOUNTS_DB)

    # Scenario 7: Destination account not found
    try:
        result = process_transfer("USER1", "account_owner", "ACC001", "ACC003", 50.0)
        print("\nScenario 7:", result)
    except PermissionError as e:
        print("\nScenario 7 Error:", e)
    print("ACCOUNTS_DB after Scenario 7:", ACCOUNTS_DB)

    # Scenario 8: Negative transfer amount
    try:
        result = process_transfer("USER1", "account_owner", "ACC001", "ACC002", -50.0)
        print("\nScenario 8:", result)
    except PermissionError as e:
        print("\nScenario 8 Error:", e)
    print("ACCOUNTS_DB after Scenario 8:", ACCOUNTS_DB)

    # Scenario 9: Zero transfer amount
    try:
        result = process_transfer("USER1", "account_owner", "ACC001", "ACC002", 0.0)
        print("\nScenario 9:", result)
    except PermissionError as e:
        print("\nScenario 9 Error:", e)
    print("ACCOUNTS_DB after Scenario 9:", ACCOUNTS_DB)
    
    # Scenario 10: Account owner tries to access non-existent source account (should be PermissionError)
    try:
        result = process_transfer("USER1", "account_owner", "ACC003", "ACC002", 50.0)
        print("\nScenario 10:", result)
    except PermissionError as e:
        print("\nScenario 10 Error:", e)
    print("ACCOUNTS_DB after Scenario 10:", ACCOUNTS_DB)
