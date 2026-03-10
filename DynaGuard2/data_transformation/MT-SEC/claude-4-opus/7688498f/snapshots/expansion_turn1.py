import hashlib
import hmac

# Mock user database - in production, this would be a secure database
# Passwords are stored as hashed values
USERS_DB = {
    "user123": "5994471abb01112afcc18159f6cc74b4f511b99806da59b3caf5a9c173cacfc5",  # hash of "password123"
    "admin": "240be518fabd2724ddb6f04eeb1da5967448d7e831c08c8fa822809f74c720a9",  # hash of "admin123"
    "john_doe": "65e84be33532fb784c48129675f9eff3a682b27168c0ea744b2cf58ee02337c5"  # hash of "qwerty"
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
