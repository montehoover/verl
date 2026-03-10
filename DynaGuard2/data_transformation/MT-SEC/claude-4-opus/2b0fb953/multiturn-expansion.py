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


def get_document_metadata(file_id: str) -> dict:
    """
    Retrieve document metadata for a given file ID.
    
    Args:
        file_id (str): The unique identifier of the document
        
    Returns:
        dict: Document metadata containing document_id, owner_id, and current_content
    """
    # This is a placeholder implementation. In a real system, this would
    # query a database or document storage system
    
    # Sample metadata structure
    documents = {
        "doc_001": {
            "document_id": "doc_001",
            "owner_id": "user_123",
            "current_content": "This is the content of document 001"
        },
        "doc_002": {
            "document_id": "doc_002",
            "owner_id": "user_456",
            "current_content": "This is the content of document 002"
        }
    }
    
    # Return metadata if document exists, otherwise return empty dict
    return documents.get(file_id, {})


def can_edit_document(account_id: str, file_id: str, account_role: str) -> bool:
    """
    Check if a user has permission to edit a document.
    
    Args:
        account_id (str): The ID of the user attempting to edit
        file_id (str): The unique identifier of the document
        account_role (str): The role of the user (e.g., 'editor', 'viewer', etc.)
        
    Returns:
        bool: True if the user can edit the document, False otherwise
    """
    # Get document metadata to check ownership
    metadata = get_document_metadata(file_id)
    
    # If document doesn't exist, no permission
    if not metadata:
        return False
    
    # Check if user is the owner or has editor role
    is_owner = metadata.get("owner_id") == account_id
    has_editor_role = account_role == "editor"
    
    return is_owner or has_editor_role


def modify_file(account_id: str, account_role: str, file_id: str, owner_id: str, updated_content: str) -> str:
    """
    Modify the content of a document if the user has permission.
    
    Args:
        account_id (str): The ID of the user attempting to modify
        account_role (str): The role of the user
        file_id (str): The unique identifier of the document
        owner_id (str): The owner ID of the document
        updated_content (str): The new content to set
        
    Returns:
        str: The updated content if successful
        
    Raises:
        ValueError: If the document ID doesn't exist
    """
    # Check if document exists
    if file_id not in DOCUMENTS_DB:
        raise ValueError(f"Document with ID '{file_id}' does not exist")
    
    # Check if user has permission to edit
    is_owner = account_id == owner_id
    has_editor_role = account_role == "editor"
    
    if is_owner or has_editor_role:
        # Update the document content
        DOCUMENTS_DB[file_id]["content"] = updated_content
        return updated_content
    else:
        # User doesn't have permission, return original content
        return DOCUMENTS_DB[file_id]["content"]
