import hashlib
import hmac
import secrets

# Simulated user database (in production, this would be a secure database)
# Passwords are stored as salted hashes, not plain text
USER_DATABASE = {
    "user123": {
        "password_hash": "5994471abb01112afcc18159f6cc74b4f511b99806da59b3caf5a9c173cacfc5",
        "salt": "random_salt_1"
    },
    "user456": {
        "password_hash": "65e84be33532fb784c48129675f9eff3a682b27168c0ea744b2cf58ee02337c5",
        "salt": "random_salt_2"
    }
}

# Simulated accounts database
ACCOUNTS_DB = {
    "ACC001": {"balance": 1500.00},
    "ACC002": {"balance": 2500.50},
    "ACC003": {"balance": 100.00}
}

def hash_password(password: str, salt: str) -> str:
    """Hash a password with a salt using SHA-256"""
    return hashlib.sha256((password + salt).encode()).hexdigest()

def authenticate_user(user_id: str, password: str) -> bool:
    """
    Authenticate a user based on user_id and password.
    
    Args:
        user_id (str): The user's identifier
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
    Update the account balance by the specified amount.
    
    Args:
        account_number (str): The account number to update
        amount (float): The amount to add (positive) or subtract (negative)
        
    Returns:
        bool: True if update was successful, False otherwise
    """
    # Check if account exists
    if account_number not in ACCOUNTS_DB:
        return False
    
    # Get current balance
    current_balance = ACCOUNTS_DB[account_number]["balance"]
    
    # Calculate new balance
    new_balance = current_balance + amount
    
    # Check if sufficient funds (balance cannot go negative)
    if new_balance < 0:
        return False
    
    # Update the balance
    ACCOUNTS_DB[account_number]["balance"] = new_balance
    
    return True
