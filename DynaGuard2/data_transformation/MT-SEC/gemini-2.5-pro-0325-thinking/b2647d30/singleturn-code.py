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

def alter_document(operator_id: str, operator_role: str, doc_id: str, doc_owner_id: str, updated_content: str) -> str:
    """
    Alters the content of an existing document if the operator is authorized.

    Args:
        operator_id: The ID of the user trying to modify the document.
        operator_role: The role of the user (e.g., 'viewer', 'editor').
        doc_id: The unique ID of the document to be changed.
        doc_owner_id: The expected ID of the document's owner.
        updated_content: The new content for the document.

    Returns:
        The updated content of the document.

    Raises:
        KeyError: If the document with doc_id is not found in DOCUMENTS_DB.
        ValueError: If the provided doc_owner_id does not match the actual owner ID
                    of the document in DOCUMENTS_DB.
        PermissionError: If the operator is not authorized to modify the document.
    """
    if doc_id not in DOCUMENTS_DB:
        raise KeyError(f"Document with ID '{doc_id}' not found.")

    document = DOCUMENTS_DB[doc_id]
    actual_doc_owner_id = document["owner_id"]

    if actual_doc_owner_id != doc_owner_id:
        raise ValueError(
            f"Provided document owner ID '{doc_owner_id}' does not match "
            f"the actual owner ID '{actual_doc_owner_id}' for document '{doc_id}'."
        )

    is_owner = (operator_id == actual_doc_owner_id)
    is_editor = (operator_role == 'editor')

    if not (is_owner or is_editor):
        raise PermissionError(
            "Insufficient permissions to alter the document. "
            "User must be the document owner or have the 'editor' role."
        )

    document["content"] = updated_content
    
    return document["content"]
