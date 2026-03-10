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


def _is_user_authorized(initiator_id: str, initiator_role: str, withdraw_account_num: str, accounts_db: dict) -> bool:
    """Check if the initiator is authorized to perform a withdrawal.

    Args:
        initiator_id: The user ID initiating the transaction.
        initiator_role: The role of the user (e.g., 'account_owner', 
                        'bank_admin', 'customer_service').
        withdraw_account_num: The account number to withdraw funds from.
        accounts_db: A dictionary representing the accounts database,
                     structured as {account_number: {"owner_id": ..., ...}}.

    Returns:
        True if the user is authorized, False otherwise.

    Raises:
        PermissionError: If an 'account_owner' tries to withdraw from an
                         account they do not own, or if the withdraw_account
                         does not exist when checking ownership for an
                         'account_owner'.
    """
    if initiator_role in ['bank_admin', 'customer_service']:
        return True
    elif initiator_role == 'account_owner':
        if withdraw_account_num not in accounts_db:
            # This specific error is for an account_owner trying to access a non-existent account for withdrawal.
            raise PermissionError(
                f"Unauthorized: User {initiator_id} (account_owner) cannot withdraw from non-existent account {withdraw_account_num}."
            )
        if accounts_db[withdraw_account_num]['owner_id'] == initiator_id:
            return True
        else:
            # This specific error is for an account_owner trying to access an account they don't own.
            raise PermissionError(
                f"Unauthorized: User {initiator_id} (account_owner) cannot withdraw from account {withdraw_account_num} as they are not the owner."
            )
    return False # Catches unrecognized roles or other fall-through cases


def _update_balances(withdraw_account_num: str, deposit_account_num: str, transfer_value: float, accounts_db: dict) -> None:
    """Update the balances of the withdrawal and deposit accounts.

    This function directly modifies the 'balance' field of the account
    dictionaries within the provided `accounts_db`.

    Args:
        withdraw_account_num: The account number to withdraw funds from.
        deposit_account_num: The account number to deposit funds into.
        transfer_value: The non-negative amount to transfer.
        accounts_db: A dictionary representing the accounts database.
                     It is expected that both `withdraw_account_num` and
                     `deposit_account_num` are valid keys in this dictionary
                     and that `accounts_db[withdraw_account_num]` has
                     sufficient balance. This dictionary will be modified
                     in place.
    """
    accounts_db[withdraw_account_num]['balance'] -= transfer_value
    accounts_db[deposit_account_num]['balance'] += transfer_value


def shift_funds(initiator_id: str, initiator_role: str, withdraw_account: str, deposit_account: str, transfer_value: float):
    """
    Handles moving money from one account to another in a banking system.

    Args:
        initiator_id: The user ID initiating the transaction.
        initiator_role: The role of the user (e.g., 'account_owner', 'bank_admin', 'customer_service').
        withdraw_account: The account to withdraw funds from.
        deposit_account: The account to deposit funds into.
        transfer_value: The money amount to transfer.

    Returns:
        A success message confirming the transfer if the user is authorized,
        otherwise an error message for unauthorized access or other issues.

    Raises:
        PermissionError: For an unauthorized access.
        ValueError: If transfer_value is negative.
    """
    if transfer_value < 0:
        raise ValueError("Transfer value cannot be negative.")

    # Authorization check
    try:
        is_authorized = _is_user_authorized(initiator_id, initiator_role, withdraw_account, ACCOUNTS_DB)
        if not is_authorized:
            # This covers cases like unrecognized roles not handled by _is_user_authorized's specific PermissionErrors
            raise PermissionError(f"Unauthorized: Role {initiator_role} with ID {initiator_id} is not permitted to perform this transfer.")
    except PermissionError:
        # Re-raise PermissionError raised by _is_user_authorized (e.g. owner mismatch)
        raise

    # Account existence and funds check
    # Note: _is_user_authorized might have already checked withdraw_account existence for 'account_owner'
    # but a general check here is still good for other roles or if logic changes.
    if withdraw_account not in ACCOUNTS_DB:
        return f"Error: Withdraw account {withdraw_account} does not exist."
    if deposit_account not in ACCOUNTS_DB:
        return f"Error: Deposit account {deposit_account} does not exist."

    if ACCOUNTS_DB[withdraw_account]['balance'] < transfer_value:
        return f"Error: Insufficient funds in account {withdraw_account}."

    # Perform transfer
    _update_balances(withdraw_account, deposit_account, transfer_value, ACCOUNTS_DB)

    return f"Success: Transferred {transfer_value} from {withdraw_account} to {deposit_account}. Initiator: {initiator_id} ({initiator_role})."
