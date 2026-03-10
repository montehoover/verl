import logging

# Configure logging
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


def check_authorization(initiator_id: str, initiator_role: str, withdraw_account: str, accounts_db: dict) -> bool:
    """Check if the initiator is authorized to withdraw from the specified account.
    
    This function verifies whether a user has the necessary permissions to withdraw
    funds from a given account based on their role. Bank admins and customer service
    representatives have universal access, while account owners can only withdraw
    from their own accounts.
    
    Args:
        initiator_id: The unique identifier of the user initiating the transaction.
        initiator_role: The role of the user (e.g., 'account_owner', 'bank_admin', 
                       or 'customer_service').
        withdraw_account: The account number from which funds will be withdrawn.
        accounts_db: A dictionary containing account information indexed by account number.
    
    Returns:
        True if the user is authorized to perform the withdrawal, False otherwise.
    
    Raises:
        PermissionError: If the initiator_role is invalid or not recognized.
    """
    if initiator_role == 'bank_admin' or initiator_role == 'customer_service':
        return True
    
    if initiator_role == 'account_owner':
        return accounts_db[withdraw_account]['owner_id'] == initiator_id
    
    raise PermissionError(f"Invalid role: {initiator_role}")


def update_balances(withdraw_account: str, deposit_account: str, transfer_value: float, accounts_db: dict) -> tuple[float, float]:
    """Update account balances for a fund transfer.
    
    This function performs the actual transfer of funds by deducting the specified
    amount from the withdrawal account and adding it to the deposit account. It
    returns the new balances for both accounts.
    
    Args:
        withdraw_account: The account number from which funds will be withdrawn.
        deposit_account: The account number to which funds will be deposited.
        transfer_value: The amount of money to transfer between accounts.
        accounts_db: A dictionary containing account information indexed by account number.
    
    Returns:
        A tuple containing the new balances (withdraw_account_balance, deposit_account_balance).
    
    Note:
        This function modifies the accounts_db dictionary in place.
    """
    accounts_db[withdraw_account]['balance'] -= transfer_value
    accounts_db[deposit_account]['balance'] += transfer_value
    return accounts_db[withdraw_account]['balance'], accounts_db[deposit_account]['balance']


def shift_funds(initiator_id: str, initiator_role: str, withdraw_account: str, deposit_account: str, transfer_value: float) -> str:
    """Transfer funds between two bank accounts with authorization checks.
    
    This function handles the complete process of transferring money from one account
    to another, including validation of account existence, authorization verification,
    balance checking, and the actual fund transfer. It ensures that only authorized
    users can perform transfers and that accounts have sufficient funds.
    
    Args:
        initiator_id: The unique identifier of the user initiating the transaction.
        initiator_role: The role of the user (e.g., 'account_owner', 'bank_admin', 
                       or 'customer_service').
        withdraw_account: The account number from which funds will be withdrawn.
        deposit_account: The account number to which funds will be deposited.
        transfer_value: The amount of money to transfer between accounts.
    
    Returns:
        A success message confirming the transfer if completed successfully,
        or an error message describing why the transfer failed.
    
    Raises:
        PermissionError: If the user is not authorized to perform the withdrawal
                        based on their role and account ownership.
    """
    # Log transfer attempt
    logger.info(f"Transfer attempt: initiator={initiator_id}, role={initiator_role}, "
                f"from={withdraw_account}, to={deposit_account}, amount=${transfer_value:.2f}")
    
    # Check if withdrawal account exists
    if withdraw_account not in ACCOUNTS_DB:
        error_msg = f"Error: Withdrawal account {withdraw_account} not found"
        logger.error(f"Transfer failed for {initiator_id}: {error_msg}")
        return error_msg
    
    # Check if deposit account exists
    if deposit_account not in ACCOUNTS_DB:
        error_msg = f"Error: Deposit account {deposit_account} not found"
        logger.error(f"Transfer failed for {initiator_id}: {error_msg}")
        return error_msg
    
    # Check authorization
    try:
        if not check_authorization(initiator_id, initiator_role, withdraw_account, ACCOUNTS_DB):
            error_msg = f"User {initiator_id} is not authorized to withdraw from account {withdraw_account}"
            logger.warning(f"Unauthorized transfer attempt: {error_msg}")
            raise PermissionError(error_msg)
    except PermissionError as e:
        logger.error(f"Transfer failed for {initiator_id}: {str(e)}")
        raise
    
    # Check if sufficient funds
    if ACCOUNTS_DB[withdraw_account]['balance'] < transfer_value:
        error_msg = f"Error: Insufficient funds in account {withdraw_account}"
        logger.warning(f"Transfer failed for {initiator_id}: {error_msg}")
        return error_msg
    
    # Perform the transfer
    withdraw_new_balance, deposit_new_balance = update_balances(withdraw_account, deposit_account, transfer_value, ACCOUNTS_DB)
    
    success_msg = f"Successfully transferred ${transfer_value:.2f} from {withdraw_account} to {deposit_account}"
    logger.info(f"Transfer successful: initiator={initiator_id}, from={withdraw_account} (new balance: ${withdraw_new_balance:.2f}), "
                f"to={deposit_account} (new balance: ${deposit_new_balance:.2f})")
    
    return success_msg
