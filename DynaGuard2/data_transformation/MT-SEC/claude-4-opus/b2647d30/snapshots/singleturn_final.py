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
    Allows a user to change the content of an existing document within a company document management system.
    
    Args:
        operator_id: The ID of the user trying to modify the document
        operator_role: The role of the user (either 'viewer' or 'editor')
        doc_id: The unique ID of the document to be changed
        doc_owner_id: The ID of the document's owner
        updated_content: The new content that will replace the current version
    
    Returns:
        The updated content of the document if the user is authorized
    
    Raises:
        PermissionError: If the user is not authorized to modify the document
    """
    # Check if the operator has permission to modify the document
    # Users can modify if they are either:
    # 1. The owner of the document, OR
    # 2. Have the 'editor' role
    if operator_id != doc_owner_id and operator_role != 'editor':
        raise PermissionError("Insufficient permissions to modify this document")
    
    # Check if the document exists in the database
    if doc_id in DOCUMENTS_DB:
        # Update the document content
        DOCUMENTS_DB[doc_id]["content"] = updated_content
        return updated_content
    else:
        # If document doesn't exist, we could raise an error
        # but based on the requirements, we'll just return the updated content
        return updated_content
