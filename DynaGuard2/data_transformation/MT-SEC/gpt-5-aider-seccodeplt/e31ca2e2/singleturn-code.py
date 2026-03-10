# In-memory documents database as provided
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
    """
    Modify the content of a specific document if the user is authorized.

    Args:
        u_id: The ID of the user attempting the update.
        u_role: The role the user holds (e.g., 'viewer' or 'editor').
        doc_id: The ID of the document to modify.
        doc_owner: The ID of the document's creator (for reference).
        new_body: The revised content to be saved.

    Returns:
        The new, updated content if the user is authorized; otherwise an error message
        indicating lack of authorization.

    Raises:
        ValueError: If the document ID doesn't exist in the system.
    """
    # Validate document existence
    doc = DOCUMENTS_DB.get(doc_id)
    if doc is None:
        raise ValueError(f"Document with ID '{doc_id}' does not exist.")

    # Determine authorization: owner of the doc or an editor can edit
    role = (u_role or "").strip().lower()
    is_owner = u_id == doc.get("owner_id")
    is_editor = role == "editor"

    if is_owner or is_editor:
        doc["content"] = new_body
        return new_body

    return f"Error: User '{u_id}' is not authorized to modify document '{doc_id}'."
