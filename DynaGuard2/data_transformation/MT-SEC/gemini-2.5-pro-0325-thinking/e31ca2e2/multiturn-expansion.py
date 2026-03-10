# For demonstration purposes, using a hardcoded user and password.
# In a real application, this would involve checking against a database
# or another secure user store, and passwords would be hashed.
_users_credentials = {
    "admin": "password123"
}

# For demonstration purposes, using a hardcoded user roles.
# In a real application, this would involve checking against a database
# or another secure user store.
_user_roles = {
    "admin": "administrator",
    "editor1": "editor",
    "viewer1": "viewer"
}

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
        True if authentication is successful, False otherwise.
    """
    if user_id in _users_credentials and _users_credentials[user_id] == password:
        return True
    return False

def check_user_role(user_id: str) -> str:
    """
    Retrieves the role of a user.

    Args:
        user_id: The user's identifier.

    Returns:
        The user's role as a string, or an empty string if the user is not found or has no role.
    """
    return _user_roles.get(user_id, "")

def edit_doc(u_id: str, u_role: str, doc_id: str, doc_owner: str, new_body: str) -> str:
    """
    Edits a document if the user is authorized.

    Args:
        u_id: The user's identifier.
        u_role: The user's role.
        doc_id: The document's identifier.
        doc_owner: The owner of the document.
        new_body: The new content for the document.

    Returns:
        The updated content of the document.

    Raises:
        ValueError: If the document ID doesn't exist.
        PermissionError: If the user is not authorized to edit the document.
    """
    if doc_id not in DOCUMENTS_DB:
        raise ValueError(f"Document with ID '{doc_id}' not found.")

    document = DOCUMENTS_DB[doc_id]

    # Authorization check
    # Administrators and editors can edit any document.
    # Users can edit documents they own.
    is_authorized = (
        u_role == "administrator" or
        u_role == "editor" or
        (document["owner_id"] == u_id and document["owner_id"] == doc_owner) # Ensure doc_owner matches actual owner
    )

    if not is_authorized:
        raise PermissionError(f"User '{u_id}' is not authorized to edit document '{doc_id}'.")

    document["content"] = new_body
    return document["content"]
