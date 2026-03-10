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
