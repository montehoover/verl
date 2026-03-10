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

# Mock documents database
DOCUMENTS_DB = {
    "doc123": {
        "document_id": "doc123",
        "owner_id": "user111",
        "content": "Original content of doc123"
    },
    "doc456": {
        "document_id": "doc456",
        "owner_id": "user222",
        "content": "Original content of doc456"
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

def alter_document(operator_id, operator_role, doc_id, doc_owner_id, updated_content):
    """
    Alter the content of an existing document.
    
    Args:
        operator_id (str): The user's ID attempting the modification
        operator_role (str): The user's role
        doc_id (str): The document ID to modify
        doc_owner_id (str): The owner ID of the document
        updated_content (str): The new content for the document
    
    Returns:
        str: The updated content if authorized
        
    Raises:
        PermissionError: If the user is not authorized to modify the document
    """
    # Check if document exists
    if doc_id not in DOCUMENTS_DB:
        raise PermissionError("Document not found")
    
    document = DOCUMENTS_DB[doc_id]
    
    # Verify the provided owner ID matches the actual owner
    if document["owner_id"] != doc_owner_id:
        raise PermissionError("Invalid document owner")
    
    # Check if operator is the owner
    if operator_id == doc_owner_id:
        # Owner can always modify their own document
        document["content"] = updated_content
        return updated_content
    
    # Check if operator has admin role
    if operator_role == "admin":
        # Admins can modify any document
        document["content"] = updated_content
        return updated_content
    
    # For non-owners and non-admins, check document permissions
    if doc_id in DOCUMENT_PERMISSIONS:
        doc_perms = DOCUMENT_PERMISSIONS[doc_id]
        
        # Check if user is in allowed users list
        if operator_id in doc_perms.get("allowed_users", []):
            document["content"] = updated_content
            return updated_content
        
        # Check if user's role is in allowed roles list (excluding viewer role)
        allowed_roles = doc_perms.get("allowed_roles", [])
        if operator_role in allowed_roles and operator_role != "viewer":
            document["content"] = updated_content
            return updated_content
    
    # If no permissions found, deny access
    raise PermissionError("User not authorized to modify this document")
