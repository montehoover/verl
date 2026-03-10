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

def move_money(person_id: str, role: str, source_account: str, destination_account: str, transfer_amount: float) -> str:
    """
    Handles moving money from one account to another in a banking system.

    Args:
        person_id: The user ID initiating the transaction.
        role: The role of the user (e.g., 'account_owner', 'bank_admin', 'customer_service').
        source_account: The account to withdraw funds from.
        destination_account: The account to deposit funds into.
        transfer_amount: The money amount to transfer.

    Returns:
        A success message confirming the transfer if the user is authorized.

    Raises:
        PermissionError: For an unauthorized access.
        ValueError: If the source or destination account does not exist, or if the transfer amount is invalid.
                    (Note: This part is an extension for robustness, not explicitly in the original request's error handling)
    """
    authorized = False
    if role in ['bank_admin', 'customer_service']:
        authorized = True
    elif role == 'account_owner':
        if source_account in ACCOUNTS_DB and ACCOUNTS_DB[source_account]['owner_id'] == person_id:
            authorized = True
        else:
            # Check if account exists before trying to access owner_id
            if source_account not in ACCOUNTS_DB:
                raise PermissionError(f"Unauthorized: Source account {source_account} not found or access denied for account owner.")
            else: # Account exists, but owner_id does not match
                raise PermissionError(f"Unauthorized: User {person_id} is not the owner of account {source_account}.")
    
    if not authorized:
        # This case should ideally be caught by the specific role checks above,
        # but as a fallback for unknown roles or other logic paths.
        raise PermissionError(f"Unauthorized: Role '{role}' does not permit this action or ownership check failed.")

    # Basic validation for accounts and amount (not part of original request but good practice)
    if source_account not in ACCOUNTS_DB:
        raise ValueError(f"Error: Source account {source_account} does not exist.")
    if destination_account not in ACCOUNTS_DB:
        # Depending on requirements, we might allow transfers to external/new accounts.
        # For now, assume destination must exist.
        raise ValueError(f"Error: Destination account {destination_account} does not exist.")
    if transfer_amount <= 0:
        raise ValueError("Error: Transfer amount must be positive.")
    
    # Placeholder for actual money transfer logic
    # For example:
    # if ACCOUNTS_DB[source_account]['balance'] < transfer_amount:
    #     raise ValueError("Error: Insufficient funds in source account.")
    # ACCOUNTS_DB[source_account]['balance'] -= transfer_amount
    # ACCOUNTS_DB[destination_account]['balance'] += transfer_amount

    return f"Successfully transferred {transfer_amount} from {source_account} to {destination_account} by {person_id} ({role})."

if __name__ == '__main__':
    # Example Usage:
    print("--- Test Cases ---")

    # Scenario 1: Successful transfer by account owner
    try:
        result = move_money("USER1", "account_owner", "ACC001", "ACC002", 100.0)
        print(f"Test 1 (Owner Success): {result}")
    except Exception as e:
        print(f"Test 1 (Owner Success) - Error: {e}")

    # Scenario 2: Unauthorized transfer by account owner (wrong account)
    try:
        result = move_money("USER2", "account_owner", "ACC001", "ACC002", 50.0)
        print(f"Test 2 (Owner Fail - Wrong Account): {result}")
    except PermissionError as e:
        print(f"Test 2 (Owner Fail - Wrong Account) - Caught Expected Error: {e}")
    except Exception as e:
        print(f"Test 2 (Owner Fail - Wrong Account) - Error: {e}")

    # Scenario 3: Successful transfer by bank admin
    try:
        result = move_money("ADMIN007", "bank_admin", "ACC002", "ACC001", 75.0)
        print(f"Test 3 (Admin Success): {result}")
    except Exception as e:
        print(f"Test 3 (Admin Success) - Error: {e}")

    # Scenario 4: Successful transfer by customer service
    try:
        result = move_money("CS001", "customer_service", "ACC001", "ACC002", 25.0)
        print(f"Test 4 (CS Success): {result}")
    except Exception as e:
        print(f"Test 4 (CS Success) - Error: {e}")

    # Scenario 5: Unauthorized role
    try:
        result = move_money("USER3", "unknown_role", "ACC001", "ACC002", 10.0)
        print(f"Test 5 (Unknown Role Fail): {result}")
    except PermissionError as e:
        print(f"Test 5 (Unknown Role Fail) - Caught Expected Error: {e}")
    except Exception as e:
        print(f"Test 5 (Unknown Role Fail) - Error: {e}")

    # Scenario 6: Account owner trying to access non-existent source account
    try:
        result = move_money("USER1", "account_owner", "ACC999", "ACC002", 100.0)
        print(f"Test 6 (Owner Fail - Non-existent Source): {result}")
    except PermissionError as e: # As per current logic, this specific case raises PermissionError
        print(f"Test 6 (Owner Fail - Non-existent Source) - Caught Expected Error: {e}")
    except ValueError as e: # If we changed it to ValueError for non-existent accounts
        print(f"Test 6 (Owner Fail - Non-existent Source) - Caught Expected Error: {e}")
    except Exception as e:
        print(f"Test 6 (Owner Fail - Non-existent Source) - Error: {e}")
    
    # Scenario 7: Valid transfer but non-existent destination account (ValueError)
    try:
        result = move_money("USER1", "account_owner", "ACC001", "ACC999", 100.0)
        print(f"Test 7 (Non-existent Destination): {result}")
    except ValueError as e:
        print(f"Test 7 (Non-existent Destination) - Caught Expected Error: {e}")
    except Exception as e:
        print(f"Test 7 (Non-existent Destination) - Error: {e}")

    # Scenario 8: Invalid transfer amount (ValueError)
    try:
        result = move_money("USER1", "account_owner", "ACC001", "ACC002", -50.0)
        print(f"Test 8 (Invalid Amount): {result}")
    except ValueError as e:
        print(f"Test 8 (Invalid Amount) - Caught Expected Error: {e}")
    except Exception as e:
        print(f"Test 8 (Invalid Amount) - Error: {e}")

    print("\n--- Current Account Balances (if modified) ---")
    # Note: Balances are not actually modified by this version of the function.
    # If balance modification logic were added, this would show the changes.
    for acc_id, details in ACCOUNTS_DB.items():
        print(f"Account: {acc_id}, Owner: {details['owner_id']}, Balance: {details['balance']}")
