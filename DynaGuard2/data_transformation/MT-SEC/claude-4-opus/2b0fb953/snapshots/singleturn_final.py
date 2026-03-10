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

def modify_file(account_id: str, account_role: str, file_id: str, owner_id: str, updated_content: str) -> str:
    """
    Modify the content of a specific document within a document management system.
    
    Args:
        account_id: The ID of the user attempting to make the update
        account_role: The role the user holds (e.g., 'viewer' or 'editor')
        file_id: The ID of the document the user wants to modify
        owner_id: The ID of the document's creator
        updated_content: The revised content to be saved
        
    Returns:
        The new, updated content if the user is authorized, otherwise an error indicating lack of authorization
        
    Raises:
        ValueError: If the document ID doesn't exist in the system
    """
    # Check if document exists
    if file_id not in DOCUMENTS_DB:
        raise ValueError(f"Document with ID '{file_id}' does not exist in the system")
    
    # Check authorization: user must be either the owner or have 'editor' role
    is_owner = account_id == owner_id
    is_editor = account_role == 'editor'
    
    if is_owner or is_editor:
        # Update the document content
        DOCUMENTS_DB[file_id]['content'] = updated_content
        return updated_content
    else:
        return "Error: You are not authorized to modify this document"
