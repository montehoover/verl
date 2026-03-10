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


def alter_document(
    operator_id: str,
    operator_role: str,
    doc_id: str,
    doc_owner_id: str,
    updated_content: str
) -> str:
    """
    Modify the content of an existing document if the operator is authorized.

    Authorization rules:
    - The document must exist.
    - The provided doc_owner_id must match the document's actual owner.
    - The operator must have role 'editor'.
    - Only the owner (operator_id == doc_owner_id) is allowed to modify the document.

    Returns:
        str: The updated content of the document.

    Raises:
        KeyError: If the document does not exist.
        PermissionError: If the operator is not authorized to modify the document.
    """
    # Ensure the document exists
    if doc_id not in DOCUMENTS_DB:
        raise KeyError(f"Document not found: {doc_id}")

    document = DOCUMENTS_DB[doc_id]

    # Validate the owner matches the database record
    if document.get("owner_id") != doc_owner_id:
        raise PermissionError("Insufficient permissions: document owner mismatch.")

    # Validate operator privileges
    if operator_role != "editor":
        raise PermissionError("Insufficient permissions: only editors can modify documents.")

    if operator_id != doc_owner_id:
        raise PermissionError("Insufficient permissions: only the document owner can modify this document.")

    # Perform the update
    document["content"] = updated_content
    return document["content"]
