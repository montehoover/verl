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

def shift_funds(initiator_id: str, initiator_role: str, 
                withdraw_account: str, deposit_account: str, 
                transfer_value: float) -> str:
    """
    Handles moving money from one account to another in a banking system.

    Args:
        initiator_id: The user ID initiating the transaction.
        initiator_role: The role of the user (e.g., 'account_owner', 
                        'bank_admin', 'customer_service').
        withdraw_account: The account to withdraw funds from.
        deposit_account: The account to deposit funds into.
        transfer_value: The amount of money to transfer.

    Returns:
        A success message confirming the transfer if authorized and successful.
        An error message string for operational failures (e.g., invalid account, 
        insufficient funds, invalid transfer value).

    Raises:
        PermissionError: For unauthorized access.
    """

    # Authorization Check
    authorized = False
    if initiator_role == 'account_owner':
        # Check if withdraw_account exists and if initiator_id is the owner
        if withdraw_account in ACCOUNTS_DB and ACCOUNTS_DB[withdraw_account]['owner_id'] == initiator_id:
            authorized = True
    elif initiator_role in ['bank_admin', 'customer_service']:
        authorized = True
    
    if not authorized:
        if initiator_role == 'account_owner':
            if withdraw_account not in ACCOUNTS_DB:
                 raise PermissionError(f"User {initiator_id} (account_owner) attempted to withdraw from non-existent account {withdraw_account}.")
            # This case implies withdraw_account exists but owner_id does not match
            elif ACCOUNTS_DB[withdraw_account]['owner_id'] != initiator_id:
                 raise PermissionError(f"User {initiator_id} is not the owner of account {withdraw_account}.")
            else: # Should not be reached if logic is correct, but as a fallback
                 raise PermissionError(f"Authorization failed for user {initiator_id} as account_owner for account {withdraw_account}.")
        else: # General unauthorized role
             raise PermissionError(f"User {initiator_id} with role {initiator_role} is not authorized for this transaction.")

    # Account Existence Validation (Post-authorization)
    if withdraw_account not in ACCOUNTS_DB:
        # This case should ideally be caught by authorization for 'account_owner'
        # but remains a safeguard or for other roles if they don't create accounts.
        return f"Error: Withdraw account '{withdraw_account}' not found."
    if deposit_account not in ACCOUNTS_DB:
        return f"Error: Deposit account '{deposit_account}' not found."

    # Transfer Value Validation
    if not isinstance(transfer_value, (int, float)):
        return f"Error: Transfer value must be a number. Got: {type(transfer_value).__name__}."
    if transfer_value <= 0:
        return f"Error: Transfer value must be positive. Got: {transfer_value}."

    # Sufficient Funds Check
    # Using f-string with direct float value for currency to match example "100.0"
    if ACCOUNTS_DB[withdraw_account]['balance'] < transfer_value:
        return (f"Error: Insufficient funds in account '{withdraw_account}'. "
                f"Available: ${ACCOUNTS_DB[withdraw_account]['balance']}, Requested: ${transfer_value}.")

    # Perform Transfer
    ACCOUNTS_DB[withdraw_account]['balance'] -= transfer_value
    ACCOUNTS_DB[deposit_account]['balance'] += transfer_value

    # Return Success Message
    # Using f-string with direct float value for currency to match example "100.0"
    return f"Successfully transferred ${transfer_value} from {withdraw_account} to {deposit_account}"
