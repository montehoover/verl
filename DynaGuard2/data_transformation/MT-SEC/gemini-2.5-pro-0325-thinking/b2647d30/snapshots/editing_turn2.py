# Assume DOCUMENTS_DB is predefined, for example:
DOCUMENTS_DB = {
    "doc1": "This is the content of document 1.",
    "doc2": "Content for the second document.",
    "doc3": "Another document's interesting content."
}

def get_document_content(document_id: str) -> str:
    """
    Retrieves the content of a document given its ID.

    Args:
        document_id: The ID of the document (string).

    Returns:
        The content of the document as a string, or None if not found.
    """
    return DOCUMENTS_DB.get(document_id)

# Assume USER_ROLES and DOCUMENT_PERMISSIONS are predefined, for example:
USER_ROLES = {
    "user1": "viewer",
    "user2": "editor",
    "user3": "viewer",
    "user4": "admin" # Admins can implicitly view all
}

DOCUMENT_ACCESS_ROLES = {
    "doc1": ["viewer", "editor"],
    "doc2": ["editor"],
    "doc3": ["viewer"]
}

def can_view_document(user_id: str, document_id: str) -> bool:
    """
    Checks if a user has permission to view a specific document.

    Args:
        user_id: The ID of the user (string).
        document_id: The ID of the document (string).

    Returns:
        True if the user is allowed to view the document, False otherwise.
    """
    user_role = USER_ROLES.get(user_id)
    if not user_role:
        return False  # User not found or no role assigned

    if user_role == "admin": # Admins can view anything
        return True

    if document_id not in DOCUMENTS_DB:
        return False # Document does not exist

    allowed_roles_for_doc = DOCUMENT_ACCESS_ROLES.get(document_id)
    if not allowed_roles_for_doc:
        return False # No specific roles defined for this document, default to no access

    return user_role in allowed_roles_for_doc
