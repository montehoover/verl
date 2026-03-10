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
    Allows users to edit the content of a document if they are the owner or have the 'editor' role.

    Args:
        u_id: The user ID attempting the edit.
        u_role: The user role ('admin', 'editor', 'viewer').
        doc_id: The document ID.
        doc_owner: The intended owner ID of the document. (Currently unused in core logic for setting ownership,
                   but u_id is compared against the document's actual owner_id for permission).
        new_body: The new content for the document.

    Returns:
        The new content of the document if editing is successful.

    Raises:
        ValueError: If the document ID doesn't exist.
        ValueError: If the user is not authorized to edit the document.
        ValueError: If u_role is invalid and doesn't grant edit rights.
    """
    if doc_id not in DOCUMENTS_DB:
        raise ValueError(f"Document with ID '{doc_id}' does not exist.")

    document = DOCUMENTS_DB[doc_id]
    is_owner = (u_id == document["owner_id"])
    is_editor = (u_role == "editor")

    if is_owner or is_editor:
        document["content"] = new_body
        # The doc_owner parameter is available if ownership update logic is added later.
        # For now, DOCUMENTS_DB[doc_id]["owner_id"] is not changed by this function.
        return new_body
    elif u_role == "admin": # Admins who are not owners and not editors
        raise ValueError(f"User '{u_id}' with role '{u_role}' is not authorized to edit document '{doc_id}'. Only owner or 'editor' role can edit.")
    elif u_role == "viewer":
        raise ValueError(f"User role '{u_role}' is not authorized to edit document '{doc_id}'.")
    else:
        # Handles any other undefined or non-privileged roles
        raise ValueError(f"Invalid user role '{u_role}' or insufficient permissions to edit document '{doc_id}'.")
