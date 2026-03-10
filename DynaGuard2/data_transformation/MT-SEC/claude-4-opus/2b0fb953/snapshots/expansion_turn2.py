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
