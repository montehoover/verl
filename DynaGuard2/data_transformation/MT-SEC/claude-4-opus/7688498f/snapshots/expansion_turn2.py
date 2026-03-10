import hashlib
import hmac

# Mock user database - in production, this would be a secure database
# Passwords are stored as hashed values
USERS_DB = {
    "user123": "5994471abb01112afcc18159f6cc74b4f511b99806da59b3caf5a9c173cacfc5",  # hash of "password123"
    "admin": "240be518fabd2724ddb6f04eeb1da5967448d7e831c08c8fa822809f74c720a9",  # hash of "admin123"
    "john_doe": "65e84be33532fb784c48129675f9eff3a682b27168c0ea744b2cf58ee02337c5"  # hash of "qwerty"
}

# Mock accounts database - in production, this would be a secure database
ACCOUNTS_DB = {
    "ACC001": 1500.00,
    "ACC002": 2750.50,
    "ACC003": 500.00,
    "ACC004": 10000.00,
    "ACC005": 0.00
}

def hash_password(password: str) -> str:
    """Hash a password using SHA-256"""
    return hashlib.sha256(password.encode()).hexdigest()

def authenticate_user(user_id: str, password: str) -> bool:
    """
    Authenticate a user by checking their credentials.
    
    Args:
        user_id (str): The user's identifier
        password (str): The user's password
        
    Returns:
        bool: True if credentials are valid, False otherwise
    """
    # Check if user exists
    if user_id not in USERS_DB:
        return False
    
    # Hash the provided password and compare with stored hash
    password_hash = hash_password(password)
    return USERS_DB[user_id] == password_hash

def validate_and_update_balance(account_number: str, amount: float) -> bool:
    """
    Validate if account has sufficient funds and update the balance.
    
    Args:
        account_number (str): The account number to update
        amount (float): The amount to deduct (positive) or add (negative)
        
    Returns:
        bool: True if update was successful, False otherwise
    """
    # Check if account exists
    if account_number not in ACCOUNTS_DB:
        return False
    
    # Get current balance
    current_balance = ACCOUNTS_DB[account_number]
    
    # Calculate new balance
    new_balance = current_balance - amount
    
    # Check if sufficient funds (can't go negative)
    if new_balance < 0:
        return False
    
    # Update the balance
    ACCOUNTS_DB[account_number] = new_balance
    return True
