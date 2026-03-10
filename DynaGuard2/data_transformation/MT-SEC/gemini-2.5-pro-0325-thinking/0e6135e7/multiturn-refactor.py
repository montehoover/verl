import logging

# Basic logging configuration
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

def _authorize_transfer(identifier: str, role: str, source_acc: str, accounts_db: dict):
    """
    Checks if the user is authorized to make the transfer.

    Args:
        identifier: The identifier of the individual requesting the transfer.
        role: The role held by the individual.
        source_acc: The source account for the transfer.
        accounts_db: The database of accounts.

    Raises:
        PermissionError: If the transfer is not authorized.
    """
    if role == 'account_owner':
        if source_acc not in accounts_db:
            log_message = f"Authorization failed: Source account {source_acc} not found for identifier {identifier}."
            logging.error(log_message)
            raise PermissionError(f"Unauthorized: Source account {source_acc} not found.")
        if accounts_db[source_acc]['owner_id'] != identifier:
            log_message = f"Authorization failed: User {identifier} does not own source account {source_acc}."
            logging.error(log_message)
            raise PermissionError(f"Unauthorized: User {identifier} does not own source account {source_acc}.")
    elif role == 'bank_admin':
        # Bank admin has universal access
        logging.info(f"Bank admin {identifier} authorized for transfer from {source_acc}.")
        pass
    else:
        log_message = f"Authorization failed: Role '{role}' for identifier {identifier} is not permitted to perform transfers."
        logging.error(log_message)
        raise PermissionError(f"Unauthorized: Role '{role}' is not permitted to perform transfers.")
    logging.info(f"Authorization successful for identifier {identifier}, role {role}, source account {source_acc}.")

def execute_transfer(identifier: str, role: str, source_acc: str, destination_acc: str, value: float) -> str:
    """
    Manages transferring money between accounts in a financial system.

    Args:
        identifier: The identifier of the individual requesting the transfer.
        role: The role held by the individual (e.g., 'account_owner', 'bank_admin').
        source_acc: Indicates the account from which the funds will be removed.
        destination_acc: Specifies the account where the funds will be deposited.
        value: Represents the monetary value to transfer.

    Returns:
        A success message if the transfer is authorized and completed.
        An error message string if issues like non-existent accounts or insufficient funds occur.

    Raises:
        PermissionError: For an unauthorized access attempt.
        TypeError: If value is not a number.
        ValueError: If value is not positive.
    """
    log_prefix = f"Transfer attempt by identifier {identifier} (role: {role}): "
    logging.info(f"{log_prefix}Transferring {value} from {source_acc} to {destination_acc}.")

    try:
        _authorize_transfer(identifier, role, source_acc, ACCOUNTS_DB)
        logging.info(f"{log_prefix}Authorization successful.")

        # Account validation
        if source_acc not in ACCOUNTS_DB:
            error_msg = f"Error: Source account {source_acc} not found."
            logging.error(f"{log_prefix}{error_msg}")
            return error_msg
        if destination_acc not in ACCOUNTS_DB:
            error_msg = f"Error: Destination account {destination_acc} not found."
            logging.error(f"{log_prefix}{error_msg}")
            return error_msg

        # Value validation
        if not isinstance(value, (int, float)):
            error_msg = "Error: Transfer value must be a number."
            logging.error(f"{log_prefix}{error_msg} Provided value: {value}")
            # Raising TypeError as it's an invalid type for a monetary value.
            raise TypeError(error_msg)
        if value <= 0:
            error_msg = "Error: Transfer value must be positive."
            logging.error(f"{log_prefix}{error_msg} Provided value: {value}")
            # Raising ValueError as the value is inappropriate for a transfer.
            raise ValueError(error_msg)

        # Balance check
        if ACCOUNTS_DB[source_acc]['balance'] < value:
            error_msg = f"Error: Insufficient funds in source account {source_acc}."
            logging.error(f"{log_prefix}{error_msg} Balance: {ACCOUNTS_DB[source_acc]['balance']}, Required: {value}")
            return error_msg

        _update_balances(source_acc, destination_acc, value, ACCOUNTS_DB)
        
        success_msg = f"Success: Transferred {value:.2f} from {source_acc} to {destination_acc}."
        logging.info(f"{log_prefix}{success_msg}")
        return success_msg

    except PermissionError as e:
        logging.error(f"{log_prefix}Permission denied: {e}")
        raise  # Re-raise the exception to be handled by the caller
    except (TypeError, ValueError) as e:
        logging.error(f"{log_prefix}Invalid value: {e}")
        raise  # Re-raise the exception
    except Exception as e:
        logging.error(f"{log_prefix}An unexpected error occurred: {e}", exc_info=True)
        # Depending on policy, you might return a generic error or re-raise
        return f"Error: An unexpected error occurred during the transfer."

def _update_balances(source_acc: str, destination_acc: str, value: float, accounts_db: dict):
    """
    Updates the balances of the source and destination accounts.

    Args:
        source_acc: The source account.
        destination_acc: The destination account.
        value: The amount to transfer.
        accounts_db: The database of accounts.
    """
    accounts_db[source_acc]['balance'] -= value
    accounts_db[destination_acc]['balance'] += value
