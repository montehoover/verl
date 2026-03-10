import datetime
import copy

INITIAL_ACCOUNTS_DB_STATE = {
    "ACC001": {
        "account_number": "ACC001",
        "owner_id": "USER1",
        "balance": 1000.0
    },
    "ACC002": {
        "account_number": "ACC002",
        "owner_id": "USER2",
        "balance": 500.0
    },
    "ACC003": { # Adding another account for more diverse testing
        "account_number": "ACC003",
        "owner_id": "USER1", # Same owner as ACC001
        "balance": 200.0
    }
}

ACCOUNTS_DB = {} # Global, managed by test harness or main script flow

def _log_event(level: str, message: str):
    """Helper function for logging transaction events."""
    timestamp = datetime.datetime.now().isoformat()
    print(f"[{timestamp}] [{level.upper()}] {message}")

def _authorize_transfer(person_id: str, role: str, source_account_id: str, accounts_db: dict):
    """
    Checks if the user is authorized to make the transfer from the source account.
    Uses guard clauses for early exits on failed authorization.
    Raises PermissionError if not authorized.
    """
    _log_event("info", f"Authorizing transfer: User '{person_id}' (Role: {role}), Source Account: '{source_account_id}'")

    if role in ['bank_admin', 'customer_service']:
        _log_event("info", f"Authorization granted for admin/service role: {role}.")
    elif role == 'account_owner':
        # Guard clause: Check if source account exists
        if source_account_id not in accounts_db:
            err_msg = f"Unauthorized: Source account '{source_account_id}' not found for account owner '{person_id}'."
            _log_event("warning", err_msg)
            raise PermissionError(err_msg)
        
        # Guard clause: Check if person_id matches the owner_id of the source account
        account_owner_id = accounts_db[source_account_id]['owner_id']
        if account_owner_id != person_id:
            err_msg = f"Unauthorized: User '{person_id}' is not the owner of account '{source_account_id}' (Owner is '{account_owner_id}')."
            _log_event("warning", err_msg)
            raise PermissionError(err_msg)
        
        _log_event("info", f"Authorization granted: User '{person_id}' is owner of account '{source_account_id}'.")
    else:
        # Guard clause: Handle unrecognized or unauthorized roles
        err_msg = f"Unauthorized: Role '{role}' is not recognized or does not have permission for transfers."
        _log_event("warning", err_msg)
        raise PermissionError(err_msg)
    
    # If execution reaches here, all checks for the given role have passed.
    _log_event("info", f"Authorization successful for user '{person_id}'.")

def _validate_transaction_details(source_account_id: str, destination_account_id: str, amount: float, accounts_db: dict):
    """
    Validates account existence, non-identical accounts, and transfer amount.
    Raises ValueError for invalid details.
    """
    _log_event("info", f"Validating transaction: Amount {amount} from '{source_account_id}' to '{destination_account_id}'.")
    if source_account_id not in accounts_db:
        err_msg = f"Validation Error: Source account '{source_account_id}' does not exist."
        _log_event("error", err_msg)
        raise ValueError(err_msg)
    if destination_account_id not in accounts_db:
        err_msg = f"Validation Error: Destination account '{destination_account_id}' does not exist."
        _log_event("error", err_msg)
        raise ValueError(err_msg)
    if source_account_id == destination_account_id:
        err_msg = f"Validation Error: Source and destination accounts cannot be the same ('{source_account_id}')."
        _log_event("error", err_msg)
        raise ValueError(err_msg)
    if amount <= 0:
        err_msg = f"Validation Error: Transfer amount ({amount}) must be positive."
        _log_event("error", err_msg)
        raise ValueError(err_msg)
    _log_event("info", "Transaction details validated successfully.")

def _update_account_balances(source_account_id: str, destination_account_id: str, amount: float, accounts_db: dict):
    """
    Updates balances in the accounts_db.
    Checks for sufficient funds. Modifies accounts_db in-place.
    Raises ValueError if insufficient funds.
    """
    _log_event("info", f"Attempting to update balances: {amount} from '{source_account_id}' to '{destination_account_id}'.")
    
    source_account = accounts_db[source_account_id]
    destination_account = accounts_db[destination_account_id]

    if source_account['balance'] < amount:
        err_msg = f"Balance Update Error: Insufficient funds in source account '{source_account_id}'. Current balance: {source_account['balance']}, Requested amount: {amount}."
        _log_event("error", err_msg)
        raise ValueError(err_msg)
    
    source_account['balance'] -= amount
    destination_account['balance'] += amount
    
    _log_event("info", f"Balances updated successfully. Source Account '{source_account_id}': New Balance {source_account['balance']}. Destination Account '{destination_account_id}': New Balance {destination_account['balance']}.")

def move_money(person_id: str, role: str, source_account: str, destination_account: str, transfer_amount: float) -> str:
    """
    Handles moving money from one account to another in a banking system.
    Orchestrates authorization, validation, and balance updates using helper functions.
    The ACCOUNTS_DB global dictionary is used for account data.
    """
    global ACCOUNTS_DB 

    try:
        _log_event("info", f"Move money request initiated: User '{person_id}' (Role: {role}), From '{source_account}' To '{destination_account}', Amount: {transfer_amount}")

        _authorize_transfer(person_id, role, source_account, ACCOUNTS_DB)
        _validate_transaction_details(source_account, destination_account, transfer_amount, ACCOUNTS_DB)
        _update_account_balances(source_account, destination_account, transfer_amount, ACCOUNTS_DB)
        
        success_message = f"Successfully transferred {transfer_amount} from {source_account} to {destination_account} by {person_id} ({role})."
        _log_event("info", f"Transfer successful: {success_message}")
        return success_message

    except PermissionError as e:
        error_message = f"PermissionError during transfer: {e}"
        _log_event("error", error_message)
        raise 
    except ValueError as e:
        error_message = f"ValueError during transfer: {e}"
        _log_event("error", error_message)
        raise 
    except Exception as e:
        error_message = f"Unexpected critical error during transfer: {type(e).__name__} - {e}"
        _log_event("critical", error_message)
        raise RuntimeError(f"An unexpected error occurred: {e}")

if __name__ == '__main__':
    print("--- Test Cases ---")

    def run_test(test_name, person_id, role, source, dest, amount, expected_exception=None):
        global ACCOUNTS_DB
        ACCOUNTS_DB = copy.deepcopy(INITIAL_ACCOUNTS_DB_STATE) 
        
        print(f"\n--- Running Test: {test_name} ---")
        initial_balances_str = ", ".join([f"{acc}: {ACCOUNTS_DB[acc]['balance']}" for acc in sorted(ACCOUNTS_DB.keys())])
        print(f"Initial Balances: {initial_balances_str}")
        
        try:
            result = move_money(person_id, role, source, dest, amount)
            print(f"Result: {result}")
            if expected_exception:
                print(f"Error: Expected {expected_exception.__name__} but got success.")
            else:
                final_balances_str = ", ".join([f"{acc}: {ACCOUNTS_DB[acc].get('balance', 'N/A')}" for acc in sorted(ACCOUNTS_DB.keys()) if acc in ACCOUNTS_DB]) # Handle if acc might be removed
                print(f"Final Balances: {final_balances_str}")
        except Exception as e:
            print(f"Caught Exception: {type(e).__name__} - {e}")
            if expected_exception:
                if isinstance(e, expected_exception):
                    print(f"Successfully caught expected {expected_exception.__name__}.")
                else:
                    print(f"Error: Expected {expected_exception.__name__} but got {type(e).__name__}.")
            else:
                print(f"Error: Unexpected exception {type(e).__name__}.")
        print("--- Test End ---")

    # Scenario 1: Successful transfer by account owner
    run_test("Test 1 (Owner Success)", "USER1", "account_owner", "ACC001", "ACC002", 100.0)

    # Scenario 2: Unauthorized transfer by account owner (wrong account)
    run_test("Test 2 (Owner Fail - Wrong Account)", "USER2", "account_owner", "ACC001", "ACC002", 50.0, PermissionError)

    # Scenario 3: Successful transfer by bank admin
    run_test("Test 3 (Admin Success)", "ADMIN007", "bank_admin", "ACC002", "ACC001", 75.0)

    # Scenario 4: Successful transfer by customer service
    run_test("Test 4 (CS Success)", "CS001", "customer_service", "ACC001", "ACC002", 25.0)

    # Scenario 5: Unauthorized role
    run_test("Test 5 (Unknown Role Fail)", "USER3", "unknown_role", "ACC001", "ACC002", 10.0, PermissionError)

    # Scenario 6: Account owner trying to access non-existent source account (PermissionError from auth)
    run_test("Test 6 (Owner Fail - Non-existent Source)", "USER1", "account_owner", "ACC999", "ACC002", 100.0, PermissionError)
    
    # Scenario 7: Valid transfer but non-existent destination account (ValueError from validation)
    run_test("Test 7 (Non-existent Destination)", "USER1", "account_owner", "ACC001", "ACC999", 100.0, ValueError)

    # Scenario 8: Invalid transfer amount (ValueError from validation)
    run_test("Test 8 (Invalid Amount)", "USER1", "account_owner", "ACC001", "ACC002", -50.0, ValueError)

    # Scenario 9: Insufficient funds (ValueError from balance update)
    run_test("Test 9 (Insufficient Funds)", "USER1", "account_owner", "ACC001", "ACC002", 2000.0, ValueError)
    
    # Scenario 10: Transfer to self (ValueError from validation)
    run_test("Test 10 (Transfer to Self)", "USER1", "account_owner", "ACC001", "ACC001", 50.0, ValueError)

    # Scenario 11: Admin transfer from non-existent account (ValueError from validation, auth passes for admin)
    run_test("Test 11 (Admin - Non-existent Source)", "ADMIN007", "bank_admin", "ACC999", "ACC001", 50.0, ValueError)

    # Scenario 12: Account owner, source account exists, but owner ID mismatch
    run_test("Test 12 (Owner Mismatch)", "USER1", "account_owner", "ACC002", "ACC003", 50.0, PermissionError)
    
    # Scenario 13: Successful transfer exhausting source balance exactly
    run_test("Test 13 (Owner Success - Exact Balance)", "USER2", "account_owner", "ACC002", "ACC001", 500.0)

    print("\n--- All Tests Completed ---")
