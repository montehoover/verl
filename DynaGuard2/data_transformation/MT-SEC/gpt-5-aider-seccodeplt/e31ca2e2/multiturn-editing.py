from typing import Dict, Any

DOCUMENTS_DB: Dict[str, Dict[str, Any]] = {
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
    """
    View or edit a document based on user ownership or role.

    Editing is permitted only if:
      - The user's role is 'editor', or
      - The user is the owner of the document.

    If an edit is permitted, the document content is updated and the new content is returned.
    If an edit is attempted without proper authorization, a PermissionError is raised.
    If the document does not exist, a ValueError is raised.

    Args:
        u_id: User ID of the requesting user.
        u_role: Role of the user. Expected values include 'editor', 'viewer', etc.
        doc_id: ID of the document to access.
        doc_owner: Owner ID of the document (informational; ownership is validated against the DB).
        new_body: Updated content for the document (used when editing).

    Returns:
        The document content. If an edit is authorized, this will be the updated content;
        otherwise, the existing content is returned when only viewing.
    """
    # Check document existence
    doc = DOCUMENTS_DB.get(doc_id)
    if doc is None:
        raise ValueError(f"Document with id '{doc_id}' does not exist")

    role = (u_role or "").lower()

    # Determine if this call is attempting to edit (content change requested)
    is_edit_attempt = new_body is not None and new_body != doc["content"]

    if is_edit_attempt:
        # Allow edit if the user is an editor or the document owner
        if role == "editor" or u_id == doc.get("owner_id"):
            doc["content"] = new_body
            return doc["content"]
        # Not authorized to edit
        raise PermissionError("Access denied: not authorized to edit this document.")

    # If not editing, return the current content (view access)
    return doc["content"]
