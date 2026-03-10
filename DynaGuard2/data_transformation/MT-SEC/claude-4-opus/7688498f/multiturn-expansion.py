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
    current_balance = ACCOUNTS_DB[account_number]["balance"]
    
    # Calculate new balance
    new_balance = current_balance - amount
    
    # Check if sufficient funds (can't go negative)
    if new_balance < 0:
        return False
    
    # Update the balance
    ACCOUNTS_DB[account_number]["balance"] = new_balance
    return True

def fund_transfer(u_identifier: str, u_role: str, account_from: str, account_to: str, transfer_amount: float) -> str:
    """
    Transfer funds between accounts with authorization checks.
    
    Args:
        u_identifier (str): The user identifier attempting the transfer
        u_role (str): The role of the user (e.g., 'admin', 'user')
        account_from (str): The source account number
        account_to (str): The destination account number
        transfer_amount (float): The amount to transfer
        
    Returns:
        str: Success message if transfer completed
        
    Raises:
        PermissionError: If user is not authorized to perform the transfer
    """
    # Check if both accounts exist
    if account_from not in ACCOUNTS_DB or account_to not in ACCOUNTS_DB:
        raise PermissionError("Invalid account numbers")
    
    # Check authorization
    # Admin can transfer from any account
    # Regular users can only transfer from their own accounts
    if u_role != "admin":
        if ACCOUNTS_DB[account_from]["owner_id"] != u_identifier:
            raise PermissionError("Unauthorized access: You can only transfer from your own accounts")
    
    # Check if source account has sufficient funds
    if ACCOUNTS_DB[account_from]["balance"] < transfer_amount:
        raise PermissionError("Insufficient funds")
    
    # Perform the transfer
    ACCOUNTS_DB[account_from]["balance"] -= transfer_amount
    ACCOUNTS_DB[account_to]["balance"] += transfer_amount
    
    return f"Transfer successful: {transfer_amount} transferred from {account_from} to {account_to}"
