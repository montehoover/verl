import hashlib
import hmac
import secrets

# Simulated user database - in production, this would be a secure database
USER_DATABASE = {
    "user123": {
        "password_hash": "5e884898da28047151d0e56f8dc6292773603d0d6aabbdd62a11ef721d1542d8",  # password: "password123"
        "salt": "randomsalt123"
    },
    "admin": {
        "password_hash": "8c6976e5b5410415bde908bd4dee15dfb167a9c873fc4bb8a81f6f2ab448a918",  # password: "admin"
        "salt": "adminsalt456"
    }
}

# Simulated account database - in production, this would be a secure database
ACCOUNT_DATABASE = {
    "ACC001": {
        "balance": 1000.00,
        "owner": "user123"
    },
    "ACC002": {
        "balance": 5000.00,
        "owner": "admin"
    },
    "ACC003": {
        "balance": 250.50,
        "owner": "user123"
    }
}

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

def hash_password(password: str, salt: str) -> str:
    """Hash a password with a salt using SHA-256."""
    return hashlib.sha256((password + salt).encode()).hexdigest()

def authenticate_user(user_id: str, password: str) -> bool:
    """
    Authenticate a user by checking their credentials.
    
    Args:
        user_id (str): The user's ID
        password (str): The user's password
        
    Returns:
        bool: True if credentials are valid, False otherwise
    """
    # Check if user exists in database
    if user_id not in USER_DATABASE:
        return False
    
    # Get user data
    user_data = USER_DATABASE[user_id]
    stored_hash = user_data["password_hash"]
    salt = user_data["salt"]
    
    # Hash the provided password with the user's salt
    provided_hash = hash_password(password, salt)
    
    # Compare hashes using constant-time comparison to prevent timing attacks
    return hmac.compare_digest(provided_hash, stored_hash)

def validate_and_update_balance(account_number: str, amount: float) -> bool:
    """
    Validate and update account balance.
    
    Args:
        account_number (str): The account number
        amount (float): The amount to add (positive) or subtract (negative)
        
    Returns:
        bool: True if update was successful, False otherwise
    """
    # Check if account exists
    if account_number not in ACCOUNT_DATABASE:
        return False
    
    # Get current balance
    current_balance = ACCOUNT_DATABASE[account_number]["balance"]
    
    # Calculate new balance
    new_balance = current_balance + amount
    
    # Check if withdrawal would result in negative balance
    if new_balance < 0:
        return False
    
    # Update balance
    ACCOUNT_DATABASE[account_number]["balance"] = round(new_balance, 2)
    
    return True

def move_funds(u_id: str, u_role: str, source_account: str, destination_account: str, money: float) -> str:
    """
    Transfer funds between accounts.
    
    Args:
        u_id (str): User ID performing the transfer
        u_role (str): User role (e.g., 'admin', 'user')
        source_account (str): Source account number
        destination_account (str): Destination account number
        money (float): Amount to transfer
        
    Returns:
        str: Success message if transfer completed
        
    Raises:
        PermissionError: If user is not authorized to perform the transfer
    """
    # Check if both accounts exist
    if source_account not in ACCOUNTS_DB or destination_account not in ACCOUNTS_DB:
        raise PermissionError("Invalid account(s)")
    
    # Get account details
    source_acc = ACCOUNTS_DB[source_account]
    dest_acc = ACCOUNTS_DB[destination_account]
    
    # Check authorization: user can only transfer from their own account unless admin
    if u_role != "admin" and source_acc["owner_id"] != u_id:
        raise PermissionError("Unauthorized: You can only transfer from your own account")
    
    # Check if source account has sufficient funds
    if source_acc["balance"] < money:
        raise PermissionError("Insufficient funds")
    
    # Check if amount is valid
    if money <= 0:
        raise PermissionError("Transfer amount must be positive")
    
    # Perform the transfer
    source_acc["balance"] -= money
    dest_acc["balance"] += money
    
    # Round to avoid floating point issues
    source_acc["balance"] = round(source_acc["balance"], 2)
    dest_acc["balance"] = round(dest_acc["balance"], 2)
    
    return f"Successfully transferred ${money:.2f} from {source_account} to {destination_account}"
