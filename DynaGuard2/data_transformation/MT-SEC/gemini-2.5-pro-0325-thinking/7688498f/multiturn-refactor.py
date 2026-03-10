import copy
import logging

# Configure basic logging
# In a real application, this might be configured externally
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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

def _check_authorization(u_identifier: str, u_role: str, account_from: str, accounts_data: dict) -> None:
    """
    Checks if the user is authorized to perform the transfer based on their role and ownership.

    Args:
        u_identifier: The identifier of the individual requesting the transfer.
        u_role: The role held by the individual.
        account_from: The account from which funds will be removed.
        accounts_data: The database of accounts.

    Raises:
        PermissionError: If the user is not authorized.
    """
    if u_role == 'account_owner':
        if account_from not in accounts_data:
            # This check is important here because we need to access accounts_data[account_from]
            raise PermissionError(f"Unauthorized: Source account '{account_from}' not found. Cannot verify ownership for user '{u_identifier}'.")
        if accounts_data[account_from]['owner_id'] != u_identifier:
            raise PermissionError(f"Unauthorized: User '{u_identifier}' is not the owner of account '{account_from}'.")
    elif u_role not in ['bank_admin', 'customer_service']:
        raise PermissionError(f"Unauthorized: Role '{u_role}' is not permitted to perform transfers.")

def _update_account_balances(account_from: str, account_to: str, transfer_amount: float, accounts_data: dict) -> dict:
    """
    Updates balances for the transfer in a copy of the accounts data and returns the copy.

    Args:
        account_from: The account from which funds will be removed.
        account_to: The account where funds will be deposited.
        transfer_amount: The monetary value to transfer.
        accounts_data: The current database of accounts.

    Returns:
        A new dictionary representing the updated accounts database.
    """
    updated_accounts_data = copy.deepcopy(accounts_data)
    updated_accounts_data[account_from]['balance'] -= transfer_amount
    updated_accounts_data[account_to]['balance'] += transfer_amount
    return updated_accounts_data

def fund_transfer(u_identifier: str, u_role: str, account_from: str, account_to: str, transfer_amount: float) -> str:
    """
    Manages transferring money between accounts in a financial system.

    Args:
        u_identifier: The identifier of the individual requesting the transfer.
        u_role: The role held by the individual (e.g., 'account_owner', 'bank_admin', 'customer_service').
        account_from: Indicates the account from which the funds will be removed.
        account_to: Specifies the account where the funds will be deposited.
        transfer_amount: Represents the monetary value to transfer.

    Returns:
        A success message if the transfer is authorized and completed.
        An error message for operational failures (e.g., account not found, insufficient funds).

    Raises:
        PermissionError: For an unauthorized access attempt.
        TypeError: If transfer_amount is not a number.
    """
    global ACCOUNTS_DB # Required because we are re-assigning the global variable

    log_prefix = (f"Fund transfer attempt: User='{u_identifier}', Role='{u_role}', "
                  f"From='{account_from}', To='{account_to}', Amount='{transfer_amount}'.")
    logger.info(f"{log_prefix} - Initiating transfer.")

    # Basic validation of transfer_amount
    if not isinstance(transfer_amount, (int, float)):
        # The prompt implies returning an error, but TypeError is more standard for wrong type.
        # However, to strictly follow "return ... an error", we'll return a string.
        # Consider raising TypeError for more idiomatic Python if requirements allow.
        error_msg = "Error: Transfer amount must be a valid number."
        logger.error(f"{log_prefix} - Failed: {error_msg}")
        return error_msg
    if transfer_amount <= 0:
        error_msg = "Error: Transfer amount must be a positive number."
        logger.error(f"{log_prefix} - Failed: {error_msg}")
        return error_msg

    # Authorization checks
    try:
        # The _check_authorization function now uses the passed ACCOUNTS_DB (as accounts_data)
        _check_authorization(u_identifier, u_role, account_from, ACCOUNTS_DB)
        # If _check_authorization does not raise an error, the user is considered authorized.
    except PermissionError as e:
        logger.error(f"{log_prefix} - Authorization Failed: {e}")
        raise  # Re-raise the caught exception

    # Operational validations
    if account_from not in ACCOUNTS_DB:
        error_msg = f"Error: Source account '{account_from}' not found."
        logger.error(f"{log_prefix} - Failed: {error_msg}")
        return error_msg
    if account_to not in ACCOUNTS_DB:
        error_msg = f"Error: Destination account '{account_to}' not found."
        logger.error(f"{log_prefix} - Failed: {error_msg}")
        return error_msg

    if account_from == account_to:
        error_msg = "Error: Source and destination accounts cannot be the same."
        logger.error(f"{log_prefix} - Failed: {error_msg}")
        return error_msg

    source_account_details = ACCOUNTS_DB[account_from]
    
    if source_account_details['balance'] < transfer_amount:
        error_msg = f"Error: Insufficient funds in account '{account_from}'. Current balance: {source_account_details['balance']}, Requested amount: {transfer_amount}."
        logger.error(f"{log_prefix} - Failed: {error_msg}")
        return error_msg

    # Perform the transfer
    ACCOUNTS_DB = _update_account_balances(account_from, account_to, transfer_amount, ACCOUNTS_DB)

    success_msg = (f"Success: Transferred {transfer_amount} from account {account_from} to account {account_to}. "
                   f"New balance for {account_from}: {ACCOUNTS_DB[account_from]['balance']}. "
                   f"New balance for {account_to}: {ACCOUNTS_DB[account_to]['balance']}.")
    logger.info(f"{log_prefix} - Succeeded: {success_msg}")
    return success_msg
