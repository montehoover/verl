# In-memory mock database of documents
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
    - The operator can modify the document if they are the document owner.
    - The operator can also modify the document if their role is 'editor'.
    - If the provided doc_owner_id does not match the stored owner of the document,
      the operation is treated as unauthorized.

    Parameters:
        operator_id (str): ID of the user attempting the modification.
        operator_role (str): Role of the user ('viewer' or 'editor').
        doc_id (str): Unique ID of the document to modify.
        doc_owner_id (str): The owner ID of the document (should match what's stored).
        updated_content (str): The new content to replace the document's current content.

    Returns:
        str: The updated content of the document on success.

    Raises:
        KeyError: If the document does not exist.
        PermissionError: If the operator lacks permission or provided owner ID mismatches.
    """
    # Ensure the document exists
    if doc_id not in DOCUMENTS_DB:
        raise KeyError(f"Document with id '{doc_id}' not found.")

    record = DOCUMENTS_DB[doc_id]
    actual_owner_id = record.get("owner_id")

    # Security check: provided owner ID must match the actual owner
    if doc_owner_id != actual_owner_id:
        raise PermissionError("Provided doc_owner_id does not match the document owner.")

    # Authorization: owner or editor can modify
    is_authorized = (operator_id == actual_owner_id) or (operator_role == "editor")
    if not is_authorized:
        raise PermissionError("Insufficient permissions to modify the document.")

    # Perform update
    record["content"] = updated_content
    return record["content"]
