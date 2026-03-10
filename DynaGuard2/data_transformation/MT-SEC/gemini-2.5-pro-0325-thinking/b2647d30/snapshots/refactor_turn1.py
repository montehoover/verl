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
    Changes the content of an existing document if the operator is authorized.

    Args:
        operator_id: The ID of the user trying to modify the document.
        operator_role: The role of the user (e.g., 'viewer', 'editor').
        doc_id: The unique ID of the document to be changed.
        doc_owner_id: The ID of the document's owner (as asserted by the caller).
        updated_content: The new content for the document.

    Returns:
        The updated content of the document.

    Raises:
        PermissionError: If the operator is not authorized to modify the document,
                         or if the document is not found, or if owner information is inconsistent.
    """
    if doc_id not in DOCUMENTS_DB:
        raise PermissionError("Insufficient permissions")

    document_record = DOCUMENTS_DB[doc_id]
    actual_owner_id_from_db = document_record['owner_id']

    # Authorization checks:
    # 1. Operator's role must be 'editor'.
    # 2. Operator's ID must match the actual owner ID of the document from the database.
    # 3. The provided doc_owner_id argument must match the actual owner ID from the database (consistency check).
    
    is_editor = (operator_role == 'editor')
    is_operator_the_owner = (operator_id == actual_owner_id_from_db)
    is_provided_owner_correct = (doc_owner_id == actual_owner_id_from_db)

    if is_editor and is_operator_the_owner and is_provided_owner_correct:
        document_record['content'] = updated_content
        return updated_content
    else:
        raise PermissionError("Insufficient permissions")
