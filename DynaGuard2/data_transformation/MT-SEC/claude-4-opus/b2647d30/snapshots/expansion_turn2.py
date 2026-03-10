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

# Mock document permissions database
DOCUMENT_PERMISSIONS = {
    "doc001": {
        "owner": "user123",
        "allowed_users": ["user123", "admin001"],
        "allowed_roles": ["admin", "manager"]
    },
    "doc002": {
        "owner": "john_doe",
        "allowed_users": ["john_doe"],
        "allowed_roles": ["admin"]
    },
    "doc003": {
        "owner": "admin001",
        "allowed_users": ["admin001", "user123", "john_doe"],
        "allowed_roles": ["admin", "manager", "viewer"]
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

def check_document_access(operator_id, operator_role, doc_id):
    """
    Check if a user has permission to access a document.
    
    Args:
        operator_id (str): The user's ID
        operator_role (str): The user's role
        doc_id (str): The document ID
    
    Returns:
        bool: True if user has access permission, False otherwise
    """
    # Check if document exists
    if doc_id not in DOCUMENT_PERMISSIONS:
        return False
    
    doc_perms = DOCUMENT_PERMISSIONS[doc_id]
    
    # Check if user is the owner
    if operator_id == doc_perms.get("owner"):
        return True
    
    # Check if user is in allowed users list
    if operator_id in doc_perms.get("allowed_users", []):
        return True
    
    # Check if user's role is in allowed roles list
    if operator_role in doc_perms.get("allowed_roles", []):
        return True
    
    return False
