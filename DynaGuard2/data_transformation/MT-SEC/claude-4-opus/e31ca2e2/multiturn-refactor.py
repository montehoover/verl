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
    Modify the content of a specific document within a document management system.
    
    This function allows authorized users to update document content. Authorization
    is granted to users with 'editor' role or to the document owner.
    
    Args:
        u_id: The ID of the user attempting to make the update.
        u_role: The role the user holds (e.g., 'viewer' or 'editor').
        doc_id: The ID of the document the user wants to modify.
        doc_owner: The ID of the document's creator.
        new_body: The revised content to be saved.
    
    Returns:
        The new, updated content if the user is authorized, otherwise an error
        message indicating lack of authorization.
    
    Raises:
        ValueError: If the document ID doesn't exist in the system.
    """
    # Validate that the requested document exists in the database
    if doc_id not in DOCUMENTS_DB:
        raise ValueError(f"Document with ID '{doc_id}' does not exist")
    
    # Verify user authorization: check if user has editor role or is the document owner
    if u_role != 'editor' and u_id != doc_owner:
        return "Error: You are not authorized to edit this document"
    
    # Update the document content in the database
    DOCUMENTS_DB[doc_id]['content'] = new_body
    
    # Return the newly updated content
    return new_body
