"""Document editing utilities for a simple in-memory document store.

This module defines a function that updates document content after applying
basic authorization checks. It uses a simple in-memory dictionary to
represent a document database.
"""

DOCUMENTS_DB = {
    "doc123": {
        "document_id": "doc123",
            "owner_id": "user111",
            "content": "Original content of doc123",
    },
    "doc456": {
        "document_id": "doc456",
            "owner_id": "user222",
            "content": "Original content of doc456",
    },
}

# Reusable error message constants for consistency.
ERROR_INVALID_OWNER = "Error: invalid document owner specified."
ERROR_UNAUTHORIZED = "Error: user is not authorized to edit this document."


def edit_doc(
    u_id: str,
    u_role: str,
    doc_id: str,
    doc_owner: str,
    new_body: str,
) -> str:
    """Modify the content of a specific document if the user is authorized.

    The function validates the document's existence, verifies the provided
    owner matches the document's owner, and enforces that only the owner with
    an 'editor' role can update the document content.

    Args:
        u_id: ID of the user attempting the update.
        u_role: Role of the user (e.g., 'viewer', 'editor').
        doc_id: ID of the document to modify.
        doc_owner: ID of the document's creator (should match the owner).
        new_body: Revised content to be saved.

    Returns:
        The updated content if the user is authorized; otherwise a string
        describing the authorization error.

    Raises:
        ValueError: If the document ID does not exist in the system.
    """
    # Map incoming parameters to more descriptive local names for readability.
    user_id = u_id
    user_role = u_role
    document_id = doc_id
    document_owner = doc_owner
    new_content = new_body

    # Guard: Ensure the document exists before proceeding.
    if document_id not in DOCUMENTS_DB:
        raise ValueError(f"Document with ID '{document_id}' does not exist.")

    # Retrieve the document record.
    document_record = DOCUMENTS_DB[document_id]

    # Guard: The provided document owner must match the stored owner.
    if document_owner != document_record.get("owner_id"):
        return ERROR_INVALID_OWNER

    # Normalize the role for consistent comparison.
    normalized_role = (user_role or "").lower()

    # Guard: User must have the 'editor' role.
    if normalized_role != "editor":
        return ERROR_UNAUTHORIZED

    # Guard: The editing user must be the owner of the document.
    if user_id != document_record.get("owner_id"):
        return ERROR_UNAUTHORIZED

    # Perform the content update.
    document_record["content"] = new_content

    # Return the new content to confirm the update.
    return document_record["content"]
