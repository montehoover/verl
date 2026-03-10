import logging
import datetime # For more structured timestamp logging if needed, though logging module handles basic timestamps

# Configure basic logging
# In a real application, this would be more sophisticated (e.g., logging to a file, rotating logs, etc.)
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


def _is_user_authorized(account_holder: str, user_role: str, withdrawal_account_id: str, db: dict) -> bool:
    """
    Checks if the user is authorized to initiate a transfer from the withdrawal account.

    Args:
        account_holder: The user ID initiating the transaction.
        user_role: The role of the user.
        withdrawal_account_id: The account ID to withdraw funds from.
        db: The accounts database.

    Returns:
        True if the user is authorized, False otherwise.
    """
    if user_role in ['bank_admin', 'customer_service']:
        return True
    if user_role == 'account_owner':
        # Ensure withdrawal_account_id exists in db before accessing its 'owner_id'
        if withdrawal_account_id in db and db[withdrawal_account_id]['owner_id'] == account_holder:
            return True
    return False


def _validate_accounts_and_amount(withdrawal_account_id: str, receiving_account_id: str, amount: float, db: dict) -> None:
    """
    Validates the fund amount and ensures accounts exist and are distinct.

    Args:
        withdrawal_account_id: The account ID to withdraw funds from.
        receiving_account_id: The account ID to deposit funds into.
        amount: The amount to transfer.
        db: The accounts database.

    Raises:
        ValueError: If amount is invalid, or accounts are not found/are the same.
    """
    if not isinstance(amount, (int, float)) or amount <= 0:
        raise ValueError("Fund amount must be a positive number.")
    if withdrawal_account_id not in db:
        raise ValueError(f"Withdrawal account '{withdrawal_account_id}' not found.")
    if receiving_account_id not in db:
        raise ValueError(f"Receiving account '{receiving_account_id}' not found.")
    if withdrawal_account_id == receiving_account_id:
        raise ValueError("Withdrawal and receiving accounts cannot be the same.")


def _process_withdrawal(account_id: str, amount: float, db: dict) -> None:
    """
    Processes the withdrawal from the specified account.

    Args:
        account_id: The account ID to withdraw from.
        amount: The amount to withdraw.
        db: The accounts database (will be modified).

    Raises:
        ValueError: If the account has insufficient funds. Note: Account existence
                    should be pre-validated by _validate_accounts_and_amount.
    """
    # Account existence is expected to be checked by _validate_accounts_and_amount
    # but a check here makes the function more robust if used independently.
    if account_id not in db:
         raise ValueError(f"Withdrawal account '{account_id}' not found during withdrawal process.") # Should ideally not happen if pre-validated
    
    if db[account_id]['balance'] < amount:
        raise ValueError(
            f"Insufficient funds in account '{account_id}'. Current balance: {db[account_id]['balance']}, requested: {amount}."
        )
    db[account_id]['balance'] -= amount


def _process_deposit(account_id: str, amount: float, db: dict) -> None:
    """
    Processes the deposit into the specified account.

    Args:
        account_id: The account ID to deposit into.
        amount: The amount to deposit.
        db: The accounts database (will be modified).

    Raises:
        ValueError: Note: Account existence should be pre-validated by
                    _validate_accounts_and_amount.
    """
    # Account existence is expected to be checked by _validate_accounts_and_amount
    # but a check here makes the function more robust if used independently.
    if account_id not in db:
        raise ValueError(f"Receiving account '{account_id}' not found during deposit process.") # Should ideally not happen if pre-validated

    db[account_id]['balance'] += amount


def send_funds(account_holder: str, user_role: str, withdrawal_account: str, receiving_account: str, fund_amount: float) -> str:
    """
    Handles moving money from one account to another in a banking system.

    Args:
        account_holder: The user ID initiating the transaction.
        user_role: The role of the user (e.g., 'account_owner', 'bank_admin', 'customer_service').
        withdrawal_account: The account ID to withdraw funds from.
        receiving_account: The account ID to deposit funds into.
        fund_amount: The amount of money to transfer.

    Returns:
        A success message confirming the transfer if the user is authorized and the transaction is valid.

    Raises:
        PermissionError: If the user is not authorized to perform the transaction.
        ValueError: If the fund amount is invalid, accounts are not found, or insufficient funds.
    """
    log_prefix = f"User '{account_holder}' ({user_role}): "
    transaction_details = f"from '{withdrawal_account}' to '{receiving_account}' for amount {fund_amount}."

    logging.info(f"{log_prefix}Attempting fund transfer {transaction_details}")

    try:
        # Step 1: Authorization check
        if not _is_user_authorized(account_holder, user_role, withdrawal_account, ACCOUNTS_DB):
            error_msg = f"{log_prefix}Authorization failed for transfer {transaction_details}"
            logging.error(error_msg)
            raise PermissionError(
                f"User '{account_holder}' with role '{user_role}' is not authorized to initiate a transfer from account '{withdrawal_account}'."
            )
        logging.info(f"{log_prefix}Authorization successful for {transaction_details}")

        # Step 2: Validate fund amount, account existence, and distinctness
        _validate_accounts_and_amount(withdrawal_account, receiving_account, fund_amount, ACCOUNTS_DB)
        logging.info(f"{log_prefix}Account and amount validation successful for {transaction_details}")

        # Step 3: Process withdrawal (includes sufficient funds check)
        _process_withdrawal(withdrawal_account, fund_amount, ACCOUNTS_DB)
        logging.info(f"{log_prefix}Withdrawal successful from '{withdrawal_account}' for amount {fund_amount}.")

        # Step 4: Process deposit
        _process_deposit(receiving_account, fund_amount, ACCOUNTS_DB)
        logging.info(f"{log_prefix}Deposit successful to '{receiving_account}' for amount {fund_amount}.")

        # Step 5: Return success message
        success_message = (
            f"Successfully transferred {fund_amount} from account '{withdrawal_account}' to account '{receiving_account}'. "
            f"Initiated by user '{account_holder}' ({user_role})."
        )
        logging.info(f"{log_prefix}Transfer successful: {success_message}")
        return success_message

    except PermissionError as pe:
        # Already logged, just re-raise
        raise pe
    except ValueError as ve:
        error_msg = f"{log_prefix}Validation or processing error during transfer {transaction_details}: {ve}"
        logging.error(error_msg)
        raise ve
    except Exception as e:
        # Catch any other unexpected errors
        error_msg = f"{log_prefix}Unexpected error during transfer {transaction_details}: {e}"
        logging.critical(error_msg, exc_info=True) # exc_info=True will log the stack trace
        raise RuntimeError(f"An unexpected error occurred during the fund transfer. Details: {e}")
