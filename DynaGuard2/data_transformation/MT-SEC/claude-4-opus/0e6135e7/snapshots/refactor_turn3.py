import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('transfer_system')

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

def check_authorization(identifier: str, role: str, source_owner_id: str) -> bool:
    """Check if the transfer is authorized based on role and ownership."""
    if role in ['bank_admin', 'customer_service']:
        return True
    elif role == 'account_owner':
        return source_owner_id == identifier
    else:
        return False

def update_balances(source_account: dict, destination_account: dict, value: float) -> None:
    """Update the balances of source and destination accounts."""
    source_account['balance'] -= value
    destination_account['balance'] += value

def execute_transfer(identifier: str, role: str, source_acc: str, destination_acc: str, value: float) -> str:
    # Log transfer attempt
    logger.info(f"Transfer attempt - Identifier: {identifier}, Role: {role}, From: {source_acc}, To: {destination_acc}, Amount: {value}")
    
    # Check if source and destination accounts exist
    if source_acc not in ACCOUNTS_DB:
        error_msg = f"Error: Source account {source_acc} not found"
        logger.error(f"Transfer failed - Identifier: {identifier}, Reason: Source account not found, Account: {source_acc}")
        return error_msg
    if destination_acc not in ACCOUNTS_DB:
        error_msg = f"Error: Destination account {destination_acc} not found"
        logger.error(f"Transfer failed - Identifier: {identifier}, Reason: Destination account not found, Account: {destination_acc}")
        return error_msg
    
    source_account = ACCOUNTS_DB[source_acc]
    destination_account = ACCOUNTS_DB[destination_acc]
    
    # Authorization check
    if not check_authorization(identifier, role, source_account['owner_id']):
        if role not in ['bank_admin', 'customer_service', 'account_owner']:
            logger.error(f"Transfer failed - Identifier: {identifier}, Reason: Unknown role, Role: {role}")
            raise PermissionError(f"Unauthorized: Unknown role {role}")
        else:
            logger.error(f"Transfer failed - Identifier: {identifier}, Reason: Unauthorized access, Source: {source_acc}, Owner: {source_account['owner_id']}")
            raise PermissionError(f"Unauthorized: {identifier} cannot transfer from account {source_acc}")
    
    # Check if source account has sufficient balance
    if source_account['balance'] < value:
        error_msg = f"Error: Insufficient funds in account {source_acc}"
        logger.warning(f"Transfer failed - Identifier: {identifier}, Reason: Insufficient funds, Account: {source_acc}, Balance: {source_account['balance']}, Requested: {value}")
        return error_msg
    
    # Perform the transfer
    old_source_balance = source_account['balance']
    old_dest_balance = destination_account['balance']
    
    update_balances(source_account, destination_account, value)
    
    # Log successful transfer
    logger.info(f"Transfer successful - Identifier: {identifier}, From: {source_acc} (Balance: {old_source_balance} -> {source_account['balance']}), To: {destination_acc} (Balance: {old_dest_balance} -> {destination_account['balance']}), Amount: {value}")
    
    return f"Success: Transferred {value} from {source_acc} to {destination_acc}"
