import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('banking.transactions')

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

def check_authorization(account_holder: str, user_role: str, withdrawal_account: str) -> bool:
    """Check if the user is authorized to withdraw from the account."""
    if user_role == 'bank_admin' or user_role == 'customer_service':
        return True
    elif user_role == 'account_owner':
        return ACCOUNTS_DB[withdrawal_account]['owner_id'] == account_holder
    else:
        return False

def withdraw_funds(account: str, amount: float) -> bool:
    """Withdraw funds from an account. Returns True if successful."""
    if ACCOUNTS_DB[account]['balance'] >= amount:
        ACCOUNTS_DB[account]['balance'] -= amount
        return True
    return False

def deposit_funds(account: str, amount: float) -> None:
    """Deposit funds into an account."""
    ACCOUNTS_DB[account]['balance'] += amount

def compose(*functions):
    """Compose functions together."""
    def inner(initial_value):
        result = initial_value
        for func in functions:
            result = func(result)
        return result
    return inner

def send_funds(account_holder: str, user_role: str, withdrawal_account: str, receiving_account: str, fund_amount: float) -> str:
    timestamp = datetime.now().isoformat()
    
    # Log transaction attempt
    logger.info(f"Transaction attempt - User: {account_holder}, Role: {user_role}, From: {withdrawal_account}, To: {receiving_account}, Amount: ${fund_amount:.2f}, Timestamp: {timestamp}")
    
    # Check if both accounts exist
    if withdrawal_account not in ACCOUNTS_DB:
        error_msg = f"Error: Withdrawal account {withdrawal_account} not found"
        logger.error(f"Transaction failed - {error_msg}, User: {account_holder}, Timestamp: {timestamp}")
        return error_msg
    if receiving_account not in ACCOUNTS_DB:
        error_msg = f"Error: Receiving account {receiving_account} not found"
        logger.error(f"Transaction failed - {error_msg}, User: {account_holder}, Timestamp: {timestamp}")
        return error_msg
    
    # Check authorization
    if not check_authorization(account_holder, user_role, withdrawal_account):
        if user_role not in ['bank_admin', 'customer_service', 'account_owner']:
            logger.error(f"Unauthorized access attempt - Unknown role: {user_role}, User: {account_holder}, Account: {withdrawal_account}, Timestamp: {timestamp}")
            raise PermissionError(f"Unauthorized access: Unknown role {user_role}")
        else:
            logger.error(f"Unauthorized access attempt - User: {account_holder} cannot access account: {withdrawal_account}, Role: {user_role}, Timestamp: {timestamp}")
            raise PermissionError(f"Unauthorized access: User {account_holder} cannot withdraw from account {withdrawal_account}")
    
    # Create a composed function for the transaction
    def execute_transaction(_):
        # Attempt withdrawal
        if not withdraw_funds(withdrawal_account, fund_amount):
            error_msg = f"Error: Insufficient funds in account {withdrawal_account}"
            logger.warning(f"Transaction failed - Insufficient funds, Account: {withdrawal_account}, Requested: ${fund_amount:.2f}, User: {account_holder}, Timestamp: {timestamp}")
            return error_msg
        
        # Deposit funds
        deposit_funds(receiving_account, fund_amount)
        
        success_msg = f"Success: Transferred ${fund_amount:.2f} from {withdrawal_account} to {receiving_account}"
        logger.info(f"Transaction successful - From: {withdrawal_account}, To: {receiving_account}, Amount: ${fund_amount:.2f}, User: {account_holder}, Role: {user_role}, Timestamp: {timestamp}")
        return success_msg
    
    # Use compose to execute the transaction
    transaction = compose(execute_transaction)
    return transaction(None)
