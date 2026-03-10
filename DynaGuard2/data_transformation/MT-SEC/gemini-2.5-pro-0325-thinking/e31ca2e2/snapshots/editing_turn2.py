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
    Allows admin users to edit the content of a document.
    Other roles do not have editing privileges via this function.

    Args:
        u_id: The user ID. (Currently unused in core logic but useful for context/logging)
        u_role: The user role ('admin', 'editor', 'viewer').
        doc_id: The document ID.
        doc_owner: The intended owner ID of the document. (Currently unused in core logic)
        new_body: The new content for the document.

    Returns:
        The new content of the document if an admin edits successfully.

    Raises:
        ValueError: If the document ID doesn't exist.
        ValueError: If u_role is 'editor' or 'viewer', as they are not authorized to edit.
        ValueError: If u_role is invalid.
    """
    if doc_id not in DOCUMENTS_DB:
        raise ValueError(f"Document with ID '{doc_id}' does not exist.")

    if u_role == "admin":
        DOCUMENTS_DB[doc_id]["content"] = new_body
        # The doc_owner parameter is available if ownership update logic is added later.
        # For now, DOCUMENTS_DB[doc_id]["owner_id"] is not changed by this function.
        return new_body
    elif u_role in ["editor", "viewer"]:
        raise ValueError(f"User role '{u_role}' is not authorized to edit document '{doc_id}'.")
    else:
        # Handles any other undefined roles
        raise ValueError(f"Invalid user role: '{u_role}'.")
