import logging

# Initialize logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Create console handler if no handlers exist
if not logger.handlers:
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

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

def update_balances(source_balance: float, target_balance: float, amount: float) -> tuple[float, float]:
    """
    Pure function to calculate updated balances after a transfer.
    
    Returns:
        tuple: (new_source_balance, new_target_balance)
    """
    return source_balance - amount, target_balance + amount

def execute_transfer(employee_id: str, role_value: str, source_acct: str, target_acct: str, total_amount: float) -> str:
    # Log transfer attempt
    logger.info(f"Transfer attempt - Employee: {employee_id}, Role: {role_value}, Source: {source_acct}, Target: {target_acct}, Amount: ${total_amount:.2f}")
    
    # Check if source and target accounts exist
    if source_acct not in ACCOUNTS_DB:
        logger.error(f"Transfer failed - Source account {source_acct} not found - Employee: {employee_id}")
        raise PermissionError(f"Source account {source_acct} not found")
    if target_acct not in ACCOUNTS_DB:
        logger.error(f"Transfer failed - Target account {target_acct} not found - Employee: {employee_id}")
        raise PermissionError(f"Target account {target_acct} not found")
    
    # Get account details
    source_account = ACCOUNTS_DB[source_acct]
    target_account = ACCOUNTS_DB[target_acct]
    
    # Guard clauses for authorization
    if role_value in ['bank_admin', 'customer_service']:
        pass  # These roles are always authorized
    elif role_value == 'account_owner' and source_account['owner_id'] == employee_id:
        pass  # Account owner can transfer from their own account
    else:
        logger.error(f"Transfer failed - Unauthorized access - Employee: {employee_id}, Role: {role_value}, Source: {source_acct}")
        raise PermissionError(f"User {employee_id} with role {role_value} is not authorized to transfer from account {source_acct}")
    
    # Check if source account has sufficient balance
    if source_account['balance'] < total_amount:
        logger.warning(f"Transfer failed - Insufficient balance - Employee: {employee_id}, Source: {source_acct}, Balance: ${source_account['balance']:.2f}, Requested: ${total_amount:.2f}")
        return f"Error: Insufficient balance in account {source_acct}"
    
    # Perform the transfer using the pure function
    new_source_balance, new_target_balance = update_balances(
        source_account['balance'],
        target_account['balance'],
        total_amount
    )
    
    source_account['balance'] = new_source_balance
    target_account['balance'] = new_target_balance
    
    logger.info(f"Transfer successful - Employee: {employee_id}, Source: {source_acct}, Target: {target_acct}, Amount: ${total_amount:.2f}, New balances - Source: ${new_source_balance:.2f}, Target: ${new_target_balance:.2f}")
    
    return f"Successfully transferred ${total_amount:.2f} from {source_acct} to {target_acct}"
