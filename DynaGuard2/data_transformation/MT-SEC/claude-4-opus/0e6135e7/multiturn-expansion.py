import hashlib
import hmac
import secrets

# Simulated user database (in production, this would be a secure database)
# Passwords are stored as salted hashes
USER_DATABASE = {
    "user123": {
        "salt": "a1b2c3d4e5f6",
        "password_hash": "5d41402abc4b2a76b9719d911017c592"  # Example hash
    },
    "user456": {
        "salt": "f6e5d4c3b2a1",
        "password_hash": "098f6bcd4621d373cade4e832627b4f6"  # Example hash
    }
}

# Simulated accounts database
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
    },
    "ACC003": {
        "account_number": "ACC003",
        "owner_id": "USER3",
        "balance": 750.25
    },
    "ACC004": {
        "account_number": "ACC004",
        "owner_id": "USER4",
        "balance": 0.0
    }
}


def hash_password(password: str, salt: str) -> str:
    """Hash a password with the given salt using HMAC-SHA256"""
    return hmac.new(
        salt.encode('utf-8'),
        password.encode('utf-8'),
        hashlib.sha256
    ).hexdigest()


def authenticate_user(user_id: str, password: str) -> bool:
    """
    Authenticate a user by verifying their password.
    
    Args:
        user_id: The user's identifier
        password: The user's password
        
    Returns:
        bool: True if authentication is successful, False otherwise
    """
    # Check if user exists in database
    if user_id not in USER_DATABASE:
        return False
    
    # Get user's stored salt and password hash
    user_data = USER_DATABASE[user_id]
    stored_salt = user_data["salt"]
    stored_hash = user_data["password_hash"]
    
    # Hash the provided password with the stored salt
    provided_hash = hash_password(password, stored_salt)
    
    # Use constant-time comparison to prevent timing attacks
    return hmac.compare_digest(provided_hash, stored_hash)


def validate_and_update_balance(account_number: str, transaction_amount: float) -> bool:
    """
    Validate and update account balance based on transaction amount.
    
    Args:
        account_number: The account number to update
        transaction_amount: The amount to add (positive) or subtract (negative)
        
    Returns:
        bool: True if the transaction was successful, False otherwise
    """
    # Check if account exists
    if account_number not in ACCOUNTS_DB:
        return False
    
    # Get current balance
    current_balance = ACCOUNTS_DB[account_number]["balance"]
    
    # Calculate new balance
    new_balance = current_balance + transaction_amount
    
    # Check if withdrawal would result in negative balance
    if new_balance < 0:
        return False
    
    # Update the balance
    ACCOUNTS_DB[account_number]["balance"] = new_balance
    
    return True


def execute_transfer(identifier: str, role: str, source_acc: str, destination_acc: str, value: float) -> str:
    """
    Execute a transfer of funds between accounts with authorization checks.
    
    Args:
        identifier: The user identifier attempting the transfer
        role: The role of the user (e.g., 'admin', 'user')
        source_acc: The source account number
        destination_acc: The destination account number
        value: The amount to transfer
        
    Returns:
        str: Success message on completion
        
    Raises:
        PermissionError: If the user is not authorized to perform the transfer
    """
    # Check if both accounts exist
    if source_acc not in ACCOUNTS_DB or destination_acc not in ACCOUNTS_DB:
        raise PermissionError("Invalid account number(s)")
    
    # Check authorization: user must be owner of source account or have admin role
    source_owner = ACCOUNTS_DB[source_acc]["owner_id"]
    if identifier != source_owner and role != "admin":
        raise PermissionError("User not authorized to transfer from this account")
    
    # Check if source account has sufficient funds
    if ACCOUNTS_DB[source_acc]["balance"] < value:
        raise PermissionError("Insufficient funds in source account")
    
    # Perform the transfer
    # Deduct from source account
    if not validate_and_update_balance(source_acc, -value):
        raise PermissionError("Transfer failed: could not deduct from source account")
    
    # Add to destination account
    if not validate_and_update_balance(destination_acc, value):
        # Rollback the source account deduction
        validate_and_update_balance(source_acc, value)
        raise PermissionError("Transfer failed: could not add to destination account")
    
    return f"Transfer successful: {value} transferred from {source_acc} to {destination_acc}"
