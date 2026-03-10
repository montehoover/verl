import logging

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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

def _is_user_authorized(uid: str, role: str, source_account: str, accounts_db: dict):
    """
    Checks if the user is authorized to perform a transfer from the source account.

    Args:
        uid: ID of the user initiating the transfer.
        role: User's role.
        source_account: Account number from which funds will be withdrawn.
        accounts_db: The database of accounts.

    Raises:
        PermissionError: If the user is not authorized.
    """
    if role == 'bank_admin':
        logging.info(f"Authorization successful for bank_admin {uid}.")
        return True # Bank admin is always authorized

    if role == 'account_owner':
        if source_account not in accounts_db:
            logging.warning(f"Authorization failed: Source account {source_account} not found for user {uid} (account_owner).")
            raise PermissionError(f"User {uid} is not authorized to transfer from account {source_account} (account not found).")
        if accounts_db[source_account]['owner_id'] == uid:
            logging.info(f"Authorization successful for account_owner {uid} for account {source_account}.")
            return True
        else:
            logging.warning(f"Authorization failed: User {uid} (account_owner) does not own source account {source_account}.")
            raise PermissionError(f"User {uid} is not authorized to transfer from account {source_account} (not owner).")
    
    # Any other role or condition not met above means not authorized for transfer
    logging.warning(f"Authorization failed: User {uid} with role {role} has insufficient privileges for transfer.")
    raise PermissionError(f"User {uid} with role {role} is not authorized to perform this transfer.")

def _check_sufficient_funds(source_account_balance: float, transfer_amount: float) -> str | None:
    """
    Verifies if the source account has sufficient funds for the transfer.

    Args:
        source_account_balance: The balance of the source account.
        transfer_amount: The amount to be transferred.

    Returns:
        An error message string if funds are insufficient, otherwise None.
    """
    if source_account_balance < transfer_amount:
        return "Error: Insufficient funds in source account."
    return None

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
    try:
        _is_user_authorized(uid, role, source_account, ACCOUNTS_DB)
    except PermissionError as e:
        logging.error(f"Transfer failed for user {uid}: {e}")
        raise  # Re-raise the caught permission error

    # Account and Funds Validation
    if source_account not in ACCOUNTS_DB:
        # This check is somewhat redundant if _is_user_authorized for 'account_owner' already checked existence.
        # However, it's a good safeguard, especially for 'bank_admin' or if auth logic changes.
        message = f"Error: Source account {source_account} not found."
        logging.error(f"Transfer attempt by {uid} ({role}): {message}")
        return message
    if destination_account not in ACCOUNTS_DB:
        message = f"Error: Destination account {destination_account} not found."
        logging.error(f"Transfer attempt by {uid} ({role}) from {source_account} to {destination_account}: {message}")
        return message

    if not isinstance(transfer_amount, (int, float)) or transfer_amount <= 0:
        message = "Error: Transfer amount must be a positive number."
        logging.error(f"Transfer attempt by {uid} ({role}) from {source_account} to {destination_account} with amount {transfer_amount}: {message}")
        return message

    funds_check_error = _check_sufficient_funds(ACCOUNTS_DB[source_account]['balance'], transfer_amount)
    if funds_check_error:
        message = f"{funds_check_error} {source_account}."
        logging.error(f"Transfer attempt by {uid} ({role}) from {source_account} to {destination_account} for {transfer_amount:.2f}: {message}")
        return message

    # Perform Transfer
    ACCOUNTS_DB[source_account]['balance'] -= transfer_amount
    ACCOUNTS_DB[destination_account]['balance'] += transfer_amount

    success_message = f"Transfer of {transfer_amount:.2f} from {source_account} to {destination_account} successful. Initiated by {uid} ({role})."
    logging.info(success_message)
    return success_message

if __name__ == '__main__':
    # Example Usage and Testing
    print("Initial ACCOUNTS_DB:", ACCOUNTS_DB)

    # Scenario 1: Successful transfer by account owner
    try:
        result = process_transfer("USER1", "account_owner", "ACC001", "ACC002", 100.0)
        print("\nScenario 1:", result)
    except PermissionError as e:
        logging.error(f"Scenario 1 caught PermissionError: {e}")
        print("\nScenario 1 Error:", e)
    print("ACCOUNTS_DB after Scenario 1:", ACCOUNTS_DB)

    # Scenario 2: Successful transfer by bank admin
    try:
        result = process_transfer("ADMIN_USER", "bank_admin", "ACC002", "ACC001", 50.0)
        print("\nScenario 2:", result)
    except PermissionError as e:
        logging.error(f"Scenario 2 caught PermissionError: {e}")
        print("\nScenario 2 Error:", e)
    print("ACCOUNTS_DB after Scenario 2:", ACCOUNTS_DB)

    # Scenario 3: Unauthorized transfer attempt by account owner (wrong account)
    try:
        result = process_transfer("USER1", "account_owner", "ACC002", "ACC001", 50.0)
        print("\nScenario 3:", result)
    except PermissionError as e:
        logging.error(f"Scenario 3 caught PermissionError: {e}")
        print("\nScenario 3 Error:", e)
    print("ACCOUNTS_DB after Scenario 3:", ACCOUNTS_DB)

    # Scenario 4: Unauthorized role
    try:
        result = process_transfer("USER_CS", "customer_service", "ACC001", "ACC002", 50.0)
        print("\nScenario 4:", result)
    except PermissionError as e:
        logging.error(f"Scenario 4 caught PermissionError: {e}")
        print("\nScenario 4 Error:", e)
    print("ACCOUNTS_DB after Scenario 4:", ACCOUNTS_DB)

    # Scenario 5: Insufficient funds
    try:
        result = process_transfer("USER1", "account_owner", "ACC001", "ACC002", 10000.0)
        print("\nScenario 5:", result)
    except PermissionError as e:
        # This scenario should not raise PermissionError, but return a message.
        # Logging for the error message is handled within process_transfer.
        print("\nScenario 5 Error (if any, expected message):", e if isinstance(e, PermissionError) else "Handled by return message")
    print("ACCOUNTS_DB after Scenario 5:", ACCOUNTS_DB)
    
    # Scenario 6: Source account not found (but user is admin, so auth passes this stage)
    try:
        result = process_transfer("ADMIN_USER", "bank_admin", "ACC003", "ACC001", 50.0)
        print("\nScenario 6:", result)
    except PermissionError as e:
        # This scenario should not raise PermissionError, but return a message.
        print("\nScenario 6 Error (if any, expected message):", e if isinstance(e, PermissionError) else "Handled by return message")
    print("ACCOUNTS_DB after Scenario 6:", ACCOUNTS_DB)

    # Scenario 7: Destination account not found
    try:
        result = process_transfer("USER1", "account_owner", "ACC001", "ACC003", 50.0)
        print("\nScenario 7:", result)
    except PermissionError as e:
        # This scenario should not raise PermissionError, but return a message.
        print("\nScenario 7 Error (if any, expected message):", e if isinstance(e, PermissionError) else "Handled by return message")
    print("ACCOUNTS_DB after Scenario 7:", ACCOUNTS_DB)

    # Scenario 8: Negative transfer amount
    try:
        result = process_transfer("USER1", "account_owner", "ACC001", "ACC002", -50.0)
        print("\nScenario 8:", result)
    except PermissionError as e:
        # This scenario should not raise PermissionError, but return a message.
        print("\nScenario 8 Error (if any, expected message):", e if isinstance(e, PermissionError) else "Handled by return message")
    print("ACCOUNTS_DB after Scenario 8:", ACCOUNTS_DB)

    # Scenario 9: Zero transfer amount
    try:
        result = process_transfer("USER1", "account_owner", "ACC001", "ACC002", 0.0)
        print("\nScenario 9:", result)
    except PermissionError as e:
        # This scenario should not raise PermissionError, but return a message.
        print("\nScenario 9 Error (if any, expected message):", e if isinstance(e, PermissionError) else "Handled by return message")
    print("ACCOUNTS_DB after Scenario 9:", ACCOUNTS_DB)
    
    # Scenario 10: Account owner tries to access non-existent source account (should be PermissionError)
    try:
        result = process_transfer("USER1", "account_owner", "ACC003", "ACC002", 50.0)
        print("\nScenario 10:", result)
    except PermissionError as e:
        logging.error(f"Scenario 10 caught PermissionError: {e}")
        print("\nScenario 10 Error:", e)
    print("ACCOUNTS_DB after Scenario 10:", ACCOUNTS_DB)
