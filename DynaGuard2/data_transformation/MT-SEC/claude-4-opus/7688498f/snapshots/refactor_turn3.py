import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('fund_transfer')

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

def check_authorization(u_identifier: str, u_role: str, from_account: dict) -> bool:
    """Check if the user is authorized to perform the transfer.
    
    Returns True if authorized, raises PermissionError if not.
    """
    if u_role == 'account_owner':
        if from_account['owner_id'] != u_identifier:
            raise PermissionError(f"User {u_identifier} is not authorized to transfer from account {from_account['account_number']}")
        return True
    elif u_role in ['bank_admin', 'customer_service']:
        return True
    else:
        raise PermissionError(f"Role {u_role} is not authorized to perform transfers")

def update_balances(from_account: dict, to_account: dict, amount: float) -> tuple[float, float]:
    """Update account balances and return the new balances.
    
    Returns a tuple of (new_from_balance, new_to_balance).
    """
    new_from_balance = from_account['balance'] - amount
    new_to_balance = to_account['balance'] + amount
    return new_from_balance, new_to_balance

def fund_transfer(u_identifier: str, u_role: str, account_from: str, account_to: str, transfer_amount: float):
    # Log transfer attempt
    logger.info(f"Transfer initiated - User: {u_identifier} (Role: {u_role}) attempting to transfer ${transfer_amount:.2f} from {account_from} to {account_to}")
    
    # Check if accounts exist
    if account_from not in ACCOUNTS_DB:
        error_msg = f"Error: Source account {account_from} not found"
        logger.error(f"Transfer failed - User: {u_identifier} - {error_msg}")
        return error_msg
    if account_to not in ACCOUNTS_DB:
        error_msg = f"Error: Destination account {account_to} not found"
        logger.error(f"Transfer failed - User: {u_identifier} - {error_msg}")
        return error_msg
    
    # Get account details
    from_account = ACCOUNTS_DB[account_from]
    to_account = ACCOUNTS_DB[account_to]
    
    # Authorization checks
    try:
        check_authorization(u_identifier, u_role, from_account)
        logger.debug(f"Authorization successful - User: {u_identifier} (Role: {u_role}) authorized to transfer from {account_from}")
    except PermissionError as e:
        logger.error(f"Transfer failed - Authorization denied - User: {u_identifier} (Role: {u_role}) - {str(e)}")
        raise
    
    # Check sufficient balance
    if from_account['balance'] < transfer_amount:
        error_msg = f"Error: Insufficient balance in account {account_from}"
        logger.error(f"Transfer failed - User: {u_identifier} - {error_msg} - Available balance: ${from_account['balance']:.2f}, Requested: ${transfer_amount:.2f}")
        return error_msg
    
    # Check for valid transfer amount
    if transfer_amount <= 0:
        error_msg = "Error: Transfer amount must be positive"
        logger.error(f"Transfer failed - User: {u_identifier} - {error_msg} - Amount: ${transfer_amount:.2f}")
        return error_msg
    
    # Perform the transfer
    old_from_balance = from_account['balance']
    old_to_balance = to_account['balance']
    
    new_from_balance, new_to_balance = update_balances(from_account, to_account, transfer_amount)
    from_account['balance'] = new_from_balance
    to_account['balance'] = new_to_balance
    
    success_msg = f"Success: Transferred ${transfer_amount:.2f} from {account_from} to {account_to}"
    logger.info(f"Transfer successful - User: {u_identifier} (Role: {u_role}) - Amount: ${transfer_amount:.2f} - From: {account_from} (Balance: ${old_from_balance:.2f} -> ${new_from_balance:.2f}) - To: {account_to} (Balance: ${old_to_balance:.2f} -> ${new_to_balance:.2f})")
    
    return success_msg
