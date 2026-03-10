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

def check_authorization(u_id: str, u_role: str, source_account_owner: str) -> bool:
    """Check if the user is authorized to perform the transfer."""
    if u_role == 'bank_admin' or u_role == 'customer_service':
        return True
    elif u_role == 'account_owner':
        return source_account_owner == u_id
    else:
        return False

def update_balances(source_balance: float, dest_balance: float, amount: float) -> tuple[float, float]:
    """Update the balances for source and destination accounts."""
    new_source_balance = source_balance - amount
    new_dest_balance = dest_balance + amount
    return new_source_balance, new_dest_balance

def move_funds(u_id: str, u_role: str, source_account: str, destination_account: str, money: float):
    # Log the transfer attempt
    logger.info(f"Transfer attempt - User: {u_id}, Role: {u_role}, From: {source_account}, To: {destination_account}, Amount: ${money:.2f}")
    
    # Check if both accounts exist
    if source_account not in ACCOUNTS_DB:
        logger.error(f"Transfer failed - Source account {source_account} not found")
        return f"Error: Source account {source_account} not found"
    if destination_account not in ACCOUNTS_DB:
        logger.error(f"Transfer failed - Destination account {destination_account} not found")
        return f"Error: Destination account {destination_account} not found"
    
    # Get account details
    source_acc = ACCOUNTS_DB[source_account]
    dest_acc = ACCOUNTS_DB[destination_account]
    
    # Check authorization
    if not check_authorization(u_id, u_role, source_acc['owner_id']):
        if u_role not in ['bank_admin', 'customer_service', 'account_owner']:
            logger.warning(f"Transfer failed - Unauthorized role: {u_role} for user: {u_id}")
            raise PermissionError(f"Role {u_role} is not authorized to perform transfers")
        else:
            logger.warning(f"Transfer failed - User {u_id} not authorized to transfer from account {source_account}")
            raise PermissionError(f"User {u_id} is not authorized to transfer from account {source_account}")
    
    # Check if source account has sufficient balance
    if source_acc['balance'] < money:
        logger.warning(f"Transfer failed - Insufficient balance in account {source_account}. Balance: ${source_acc['balance']:.2f}, Requested: ${money:.2f}")
        return f"Error: Insufficient balance in source account {source_account}"
    
    # Check if amount is valid
    if money <= 0:
        logger.warning(f"Transfer failed - Invalid amount: ${money:.2f}")
        return "Error: Transfer amount must be positive"
    
    # Perform the transfer
    new_source_balance, new_dest_balance = update_balances(source_acc['balance'], dest_acc['balance'], money)
    source_acc['balance'] = new_source_balance
    dest_acc['balance'] = new_dest_balance
    
    logger.info(f"Transfer successful - User: {u_id}, From: {source_account} (new balance: ${new_source_balance:.2f}), To: {destination_account} (new balance: ${new_dest_balance:.2f}), Amount: ${money:.2f}")
    
    return f"Successfully transferred ${money:.2f} from {source_account} to {destination_account}"
