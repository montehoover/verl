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

def check_authorization(uid: str, role: str, source_account_owner_id: str) -> bool:
    """Check if the user is authorized to perform the transfer."""
    if role == 'bank_admin' or role == 'customer_service':
        return True
    elif role == 'account_owner':
        return source_account_owner_id == uid
    else:
        return False

def check_sufficient_balance(account_balance: float, transfer_amount: float) -> bool:
    """Check if the account has sufficient balance for the transfer."""
    return account_balance >= transfer_amount

def process_transfer(uid: str, role: str, source_account: str, destination_account: str, transfer_amount: float) -> str:
    # Guard clause: Check if source account exists
    if source_account not in ACCOUNTS_DB:
        logger.warning(f"Transfer failed: Source account {source_account} not found. User: {uid}, Role: {role}")
        return f"Error: Source account {source_account} not found"
    
    # Guard clause: Check if destination account exists
    if destination_account not in ACCOUNTS_DB:
        logger.warning(f"Transfer failed: Destination account {destination_account} not found. User: {uid}, Role: {role}")
        return f"Error: Destination account {destination_account} not found"
    
    # Get account details
    source_acc = ACCOUNTS_DB[source_account]
    dest_acc = ACCOUNTS_DB[destination_account]
    
    # Guard clause: Check authorization
    if not check_authorization(uid, role, source_acc['owner_id']):
        logger.error(f"Failed authentication attempt: User {uid} with role {role} tried to transfer from account {source_account}")
        if role == 'account_owner':
            raise PermissionError(f"User {uid} is not authorized to transfer from account {source_account}")
        else:
            raise PermissionError(f"Role {role} is not authorized to perform transfers")
    
    # Guard clause: Check sufficient balance
    if not check_sufficient_balance(source_acc['balance'], transfer_amount):
        logger.warning(f"Transfer failed: Insufficient balance in account {source_account}. User: {uid}, Amount: {transfer_amount}, Available: {source_acc['balance']}")
        return f"Error: Insufficient balance in source account {source_account}"
    
    # Perform the transfer
    source_acc['balance'] -= transfer_amount
    dest_acc['balance'] += transfer_amount
    
    logger.info(f"Successful transfer: {transfer_amount} from {source_account} to {destination_account}. Initiated by: {uid} (Role: {role})")
    return f"Transfer successful: {transfer_amount} transferred from {source_account} to {destination_account}"
