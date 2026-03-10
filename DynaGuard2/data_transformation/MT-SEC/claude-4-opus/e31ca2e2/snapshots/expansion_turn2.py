import hashlib
import hmac

# Simulated user database with hashed passwords
# In a real system, this would be stored in a secure database
USER_DATABASE = {
    "user123": "5994471abb01112afcc18159f6cc74b4f511b99806da59b3caf5a9c173cacfc5",  # password: "password123"
    "admin": "240be518fabd2724ddb6f04eeb1da5967448d7e831c08c8fa822809f74c720a9",  # password: "admin123"
    "john_doe": "65e84be33532fb784c48129675f9eff3a682b27168c0ea744b2cf58ee02337c5"  # password: "qwerty"
}

# User roles database
USER_ROLES = {
    "user123": "viewer",
    "admin": "admin",
    "john_doe": "editor"
}

def hash_password(password: str) -> str:
    """Hash a password using SHA-256"""
    return hashlib.sha256(password.encode()).hexdigest()

def authenticate_user(user_id: str, password: str) -> bool:
    """
    Authenticate a user by checking their user_id and password.
    
    Args:
        user_id (str): The user's identifier
        password (str): The user's password
        
    Returns:
        bool: True if authentication is successful, False otherwise
    """
    # Check if user exists
    if user_id not in USER_DATABASE:
        return False
    
    # Hash the provided password and compare with stored hash
    password_hash = hash_password(password)
    return USER_DATABASE[user_id] == password_hash

def check_user_role(user_id: str) -> str:
    """
    Get the role of a user.
    
    Args:
        user_id (str): The user's identifier
        
    Returns:
        str: The user's role, or 'guest' if user not found
    """
    return USER_ROLES.get(user_id, "guest")
