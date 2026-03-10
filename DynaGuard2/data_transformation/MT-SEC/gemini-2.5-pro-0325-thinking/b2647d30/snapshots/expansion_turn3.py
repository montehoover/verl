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

def authenticate_user(user_id: str, password: str) -> bool:
    """
    Authenticates a user based on their user_id and password.

    Args:
        user_id: The user's identifier.
        password: The user's password.

    Returns:
        True if the credentials are correct, False otherwise.
    """
    # This is a placeholder for actual authentication logic.
    # In a real system, you would check credentials against a database
    # or another secure authentication mechanism.
    # For demonstration purposes, let's assume a hardcoded user.
    if user_id == "admin" and password == "password123":
        return True
    return False

def check_document_access(operator_id: str, operator_role: str, doc_id: str) -> bool:
    """
    Checks if a user has permission to access a specific document.

    Args:
        operator_id: The identifier of the user trying to access the document.
        operator_role: The role of the user (e.g., 'admin', 'editor', 'viewer').
        doc_id: The identifier of the document.

    Returns:
        True if the user has access, False otherwise.
    """
    # This is a placeholder for actual document access logic.
    # In a real system, you would check against a database or access control lists (ACLs).
    # For demonstration purposes, let's implement some simple rules:
    # - Admins can access any document.
    # - Editors can access documents they own (placeholder: doc_id contains operator_id).
    # - Viewers can access 'public' documents.

    if operator_role == "admin":
        return True

    if operator_role == "editor":
        # Placeholder: an editor can access documents where their ID is part of the doc_id
        # This is a simplistic check and would be more complex in a real system.
        if operator_id in doc_id:
            return True

    if operator_role == "viewer":
        # Placeholder: viewers can only access documents marked as 'public'
        if "public" in doc_id: # Assuming doc_id might contain access level info
            return True
            
    return False

def alter_document(operator_id: str, operator_role: str, doc_id: str, doc_owner_id: str, updated_content: str) -> str:
    """
    Allows an authorized user to change the content of an existing document.

    Args:
        operator_id: The identifier of the user trying to alter the document.
        operator_role: The role of the user (e.g., 'admin', 'editor').
        doc_id: The identifier of the document to be altered.
        doc_owner_id: The identifier of the owner of the document. 
                       (Note: In a real system, this might be fetched from DOCUMENTS_DB based on doc_id)
        updated_content: The new content for the document.

    Returns:
        The updated content if the user is authorized.

    Raises:
        PermissionError: If the user is not authorized to alter the document.
        ValueError: If the document ID is not found in DOCUMENTS_DB.
    """
    if doc_id not in DOCUMENTS_DB:
        raise ValueError(f"Document with ID '{doc_id}' not found.")

    # Check authorization
    can_alter = False
    if operator_role == "admin":
        can_alter = True
    elif operator_role == "editor" and DOCUMENTS_DB[doc_id]["owner_id"] == operator_id:
        # Ensuring the operator_id matches the owner_id stored in the DB for the given doc_id
        # The passed doc_owner_id argument might be redundant if we always trust the DB's owner_id
        can_alter = True
    
    if not can_alter:
        raise PermissionError(f"User '{operator_id}' with role '{operator_role}' is not authorized to alter document '{doc_id}'.")

    # Update the document content
    DOCUMENTS_DB[doc_id]["content"] = updated_content
    return updated_content
