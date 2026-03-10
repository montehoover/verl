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

# Documents database
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

def edit_doc(u_id: str, u_role: str, doc_id: str, doc_owner: str, new_body: str) -> str:
    """
    Edit a document if the user is authorized.
    
    Args:
        u_id (str): The user's identifier
        u_role (str): The user's role
        doc_id (str): The document identifier
        doc_owner (str): The document owner's identifier
        new_body (str): The new content for the document
        
    Returns:
        str: The updated content if authorized
        
    Raises:
        ValueError: If the document ID doesn't exist
    """
    # Check if document exists
    if doc_id not in DOCUMENTS_DB:
        raise ValueError(f"Document {doc_id} not found")
    
    # Check authorization
    # Admin can edit any document
    # Editor can edit any document
    # Owner can edit their own document
    # Viewer cannot edit
    if u_role == "admin" or u_role == "editor" or (u_id == doc_owner):
        DOCUMENTS_DB[doc_id]["content"] = new_body
        return new_body
    else:
        raise PermissionError(f"User {u_id} with role {u_role} is not authorized to edit document {doc_id}")
