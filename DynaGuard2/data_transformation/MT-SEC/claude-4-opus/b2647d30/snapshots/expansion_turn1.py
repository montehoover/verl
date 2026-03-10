import hashlib

# Mock user database - in production, this would connect to a real database
USERS_DB = {
    "user123": {
        "password_hash": "5e884898da28047151d0e56f8dc6292773603d0d6aabbdd62a11ef721d1542d8",  # password
        "active": True
    },
    "admin001": {
        "password_hash": "8c6976e5b5410415bde908bd4dee15dfb167a9c873fc4bb8a81f6f2ab448a918",  # admin
        "active": True
    },
    "john_doe": {
        "password_hash": "ef92b778bafe771e89245b89ecbc08a44a4e166c06659911881f383d4473e94f",  # password123
        "active": True
    }
}

def hash_password(password):
    """Hash a password using SHA-256"""
    return hashlib.sha256(password.encode()).hexdigest()

def authenticate_user(user_id, password):
    """
    Authenticate a user based on user_id and password.
    
    Args:
        user_id (str): The user's ID
        password (str): The user's password
    
    Returns:
        bool: True if credentials are valid and user is active, False otherwise
    """
    # Check if user exists in database
    if user_id not in USERS_DB:
        return False
    
    user = USERS_DB[user_id]
    
    # Check if user account is active
    if not user.get("active", False):
        return False
    
    # Hash the provided password and compare with stored hash
    password_hash = hash_password(password)
    
    return password_hash == user.get("password_hash")
