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
    # Check if the document exists
    if doc_id not in DOCUMENTS_DB:
        raise PermissionError("Document not found")
    
    # Get the document
    document = DOCUMENTS_DB[doc_id]
    
    # Check if the provided owner ID matches the actual owner
    if document["owner_id"] != doc_owner_id:
        raise PermissionError("Incorrect document owner information")
    
    # Check if the operator has permission to edit
    # Only the owner or an editor can modify the document
    if operator_role == 'viewer':
        raise PermissionError("Viewers do not have permission to modify documents")
    
    if operator_role == 'editor' and operator_id != doc_owner_id:
        # Editors can only edit if they are the owner
        raise PermissionError("Only the document owner can modify this document")
    
    # Update the document content
    DOCUMENTS_DB[doc_id]["content"] = updated_content
    
    return updated_content
