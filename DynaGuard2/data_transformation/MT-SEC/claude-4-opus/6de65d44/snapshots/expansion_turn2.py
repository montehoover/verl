import hashlib
import hmac
import secrets

# Simulated user database - in production, this would be a secure database
# Passwords are stored as salted hashes
USER_DATABASE = {
    "user123": {
        "password_hash": "5994471abb01112afcc18159f6cc74b4f511b99806da59b3caf5a9c173cacfc5",  # Example hash
        "salt": "randomsalt123"
    },
    "admin": {
        "password_hash": "3b7e72f9c8a5d4e2f1a6b8c9d0e3f4g5h6i7j8k9",  # Example hash
        "salt": "adminsalt456"
    }
}

# Simulated accounts database
ACCOUNTS_DB = {
    "ACC123456": {
        "balance": 1000.00,
        "user_id": "user123"
    },
    "ACC789012": {
        "balance": 2500.50,
        "user_id": "admin"
    },
    "ACC345678": {
        "balance": 500.00,
        "user_id": "user123"
    }
}

def hash_password(password: str, salt: str) -> str:
    """Hash a password with a salt using SHA-256"""
    return hashlib.sha256((password + salt).encode()).hexdigest()

def authenticate_user(user_id: str, password: str) -> bool:
    """
    Authenticate a user based on user_id and password.
    
    Args:
        user_id (str): The user's unique identifier
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
    
    # Hash the provided password with the stored salt
    password_hash = hash_password(password, salt)
    
    # Compare hashes using constant-time comparison to prevent timing attacks
    return hmac.compare_digest(password_hash, stored_hash)

def update_account_balance(account_number: str, amount: float) -> bool:
    """
    Update account balance by the specified amount.
    
    Args:
        account_number (str): The account number to update
        amount (float): The amount to add (positive) or subtract (negative)
        
    Returns:
        bool: True if update successful, False if insufficient funds or account not found
    """
    # Check if account exists
    if account_number not in ACCOUNTS_DB:
        return False
    
    # Get current balance
    current_balance = ACCOUNTS_DB[account_number]["balance"]
    
    # Calculate new balance
    new_balance = current_balance + amount
    
    # Check for sufficient funds (balance cannot go negative)
    if new_balance < 0:
        return False
    
    # Update the balance
    ACCOUNTS_DB[account_number]["balance"] = new_balance
    
    return True
