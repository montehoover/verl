"""
Manages document editing functionalities.

This module provides a function to edit documents based on user roles and ownership,
interacting with a predefined document database.
"""

# In-memory database simulation for documents.
# Each key is a document ID, and the value is a dictionary containing
# 'document_id', 'owner_id', and 'content'.
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

def edit_doc(u_id: str, u_role: str, doc_id: str, doc_owner: str, new_body: str) -> str:
    """Modify a document's content if the user is authorized.

    Args:
        u_id: The ID of the user attempting the update.
        u_role: The role of the user (e.g., 'viewer', 'editor').
        doc_id: The ID of the document to modify.
        doc_owner: The ID of the document's creator (as provided by the caller).
                   Note: This argument is not currently used for authorization;
                   the owner_id from DOCUMENTS_DB is used instead.
        new_body: The new content for the document.

    Returns:
        The updated document content if successful.
        An error message string if authorization fails.

    Raises:
        ValueError: If `doc_id` does not exist in `DOCUMENTS_DB`.
    """
    # First guard clause: Validate document existence.
    if doc_id not in DOCUMENTS_DB:
        raise ValueError(f"Document with ID '{doc_id}' does not exist.")

    # Retrieve document details from the database.
    document = DOCUMENTS_DB[doc_id]
    actual_db_owner_id = document["owner_id"]

    # Second guard clause: Check user authorization.
    # A user is authorized if they are an 'editor' or if their user ID
    # matches the document's owner ID stored in the database.
    user_is_editor = (u_role == 'editor')
    user_is_owner = (u_id == actual_db_owner_id)

    if not (user_is_editor or user_is_owner):
        return (f"Error: User '{u_id}' (Role: '{u_role}') is not authorized "
                f"to edit document '{doc_id}'.")

    # If all checks pass, update the document content.
    document["content"] = new_body
    return new_body
