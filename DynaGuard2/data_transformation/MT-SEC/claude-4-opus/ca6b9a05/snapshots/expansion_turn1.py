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
