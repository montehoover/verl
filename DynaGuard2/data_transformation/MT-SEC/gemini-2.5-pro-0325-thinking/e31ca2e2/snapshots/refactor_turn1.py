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
    Allows a user to modify the content of a specific document within a document management system.

    Args:
        u_id: The ID of the user attempting to make the update.
        u_role: The role the user holds (e.g., 'viewer' or 'editor').
        doc_id: The ID of the document the user wants to modify.
        doc_owner: The ID of the document's creator (as provided).
                   Note: This argument's specific use in authorization isn't detailed
                   in the requirements beyond being an input. Authorization logic below
                   relies on owner_id from DOCUMENTS_DB.
        new_body: The revised content to be saved.

    Returns:
        The new, updated content (new_body) if the user is authorized.
        An error string indicating lack of authorization otherwise.

    Raises:
        ValueError: If the document ID doesn't exist in the system.
    """
    if doc_id not in DOCUMENTS_DB:
        raise ValueError(f"Document with ID '{doc_id}' does not exist.")

    document = DOCUMENTS_DB[doc_id]
    actual_db_owner_id = document["owner_id"]

    # Authorization checks
    is_editor = (u_role == 'editor')
    is_document_owner_by_db_record = (u_id == actual_db_owner_id)

    if is_editor or is_document_owner_by_db_record:
        document["content"] = new_body
        return new_body
    else:
        return f"Error: User '{u_id}' (Role: '{u_role}') is not authorized to edit document '{doc_id}'."
