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

def check_authorization(person_id: str, role: str, source_account_owner: str) -> bool:
    """Check if a person is authorized to transfer from the source account."""
    if role == 'bank_admin':
        return True
    elif role == 'customer_service':
        return True
    elif role == 'account_owner':
        return source_account_owner == person_id
    return False

def update_account_balance(account_number: str, amount: float) -> None:
    """Update the balance of an account by the specified amount."""
    ACCOUNTS_DB[account_number]['balance'] += amount

def log_transaction(source_account: str, destination_account: str, amount: float, person_id: str, role: str) -> str:
    """Generate a detailed log message for the transaction."""
    return f"Successfully transferred ${amount:.2f} from {source_account} to {destination_account}"

def validate_accounts_exist(source_account: str, destination_account: str) -> tuple[bool, str]:
    """Validate that both source and destination accounts exist."""
    if source_account not in ACCOUNTS_DB:
        return False, f"Error: Source account {source_account} not found"
    if destination_account not in ACCOUNTS_DB:
        return False, f"Error: Destination account {destination_account} not found"
    return True, ""

def validate_sufficient_balance(account_number: str, amount: float) -> tuple[bool, str]:
    """Check if an account has sufficient balance for a transaction."""
    if ACCOUNTS_DB[account_number]['balance'] < amount:
        return False, f"Error: Insufficient balance in source account {account_number}"
    return True, ""

def move_money(person_id: str, role: str, source_account: str, destination_account: str, transfer_amount: float) -> str:
    # Validate accounts exist
    accounts_valid, error_msg = validate_accounts_exist(source_account, destination_account)
    if not accounts_valid:
        return error_msg
    
    # Get account details
    source_acc = ACCOUNTS_DB[source_account]
    
    # Check authorization
    is_authorized = check_authorization(person_id, role, source_acc['owner_id'])
    
    if not is_authorized:
        raise PermissionError(f"User {person_id} with role {role} is not authorized to transfer from account {source_account}")
    
    # Check sufficient balance
    balance_valid, balance_error = validate_sufficient_balance(source_account, transfer_amount)
    if not balance_valid:
        return balance_error
    
    # Perform the transfer
    update_account_balance(source_account, -transfer_amount)
    update_account_balance(destination_account, transfer_amount)
    
    # Log and return success message
    return log_transaction(source_account, destination_account, transfer_amount, person_id, role)
