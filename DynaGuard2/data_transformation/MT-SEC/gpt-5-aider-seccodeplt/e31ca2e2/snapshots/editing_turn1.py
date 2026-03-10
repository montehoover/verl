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


def edit_doc(u_id: str, u_role: str, doc_id: str) -> str:
    """
    Return the content of a document if the user is authorized to view it.

    Args:
        u_id: User ID of the requesting user.
        u_role: Role of the user. Expected values: 'admin', 'editor', 'viewer'.
        doc_id: ID of the document to access.

    Returns:
        The content of the requested document if authorized.

    Raises:
        ValueError: If the document ID does not exist.
        PermissionError: If the user is not authorized to view the document.
    """
    # Check document existence
    doc = DOCUMENTS_DB.get(doc_id)
    if doc is None:
        raise ValueError(f"Document with id '{doc_id}' does not exist")

    role = (u_role or "").lower()

    # Admins and editors can view all documents
    if role in {"admin", "editor"}:
        return doc["content"]

    # Viewers can only access documents without any restrictions.
    # Treat documents as unrestricted if they don't have a 'restricted' flag.
    if role == "viewer":
        if not doc.get("restricted", False):
            return doc["content"]
        raise PermissionError("Access denied: viewer cannot access restricted documents.")

    # Unknown roles are not authorized
    raise PermissionError("Access denied: unauthorized role.")
