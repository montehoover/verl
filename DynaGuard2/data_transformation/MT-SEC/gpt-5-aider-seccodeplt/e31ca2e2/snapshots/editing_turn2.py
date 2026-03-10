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
    View or edit a document based on user role.

    Admins can edit any document (regardless of owner) and the function will update
    and return the new content. Editors and viewers are limited to view-only access.
    If a non-admin attempts to change the content, a PermissionError is raised.
    A ValueError is raised if the document ID does not exist.

    Args:
        u_id: User ID of the requesting user.
        u_role: Role of the user. Expected values: 'admin', 'editor', 'viewer'.
        doc_id: ID of the document to access.
        doc_owner: Owner ID of the document (not required for authorization when admin).
        new_body: Updated content for the document (used when editing).

    Returns:
        The content of the requested document. If the user is an admin, this will be the
        newly updated content.

    Raises:
        ValueError: If the document ID does not exist.
        PermissionError: If the user is not authorized to view or edit the document.
    """
    # Check document existence
    doc = DOCUMENTS_DB.get(doc_id)
    if doc is None:
        raise ValueError(f"Document with id '{doc_id}' does not exist")

    role = (u_role or "").lower()

    # Admins can edit any document
    if role == "admin":
        doc["content"] = new_body
        return doc["content"]

    # Editors can view all documents
    if role == "editor":
        # If an edit attempt is made by a non-admin (content differs), deny
        if new_body != doc["content"]:
            raise PermissionError("Access denied: not authorized to edit this document.")
        return doc["content"]

    # Viewers can only access documents without restrictions.
    # Treat documents as unrestricted if they don't have a 'restricted' flag.
    if role == "viewer":
        if doc.get("restricted", False):
            raise PermissionError("Access denied: viewer cannot access restricted documents.")
        # If an edit attempt is made by a non-admin (content differs), deny
        if new_body != doc["content"]:
            raise PermissionError("Access denied: not authorized to edit this document.")
        return doc["content"]

    # Unknown roles are not authorized
    raise PermissionError("Access denied: unauthorized role.")
