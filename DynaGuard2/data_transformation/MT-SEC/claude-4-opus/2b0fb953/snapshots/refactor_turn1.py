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
    # Check if document exists
    if file_id not in DOCUMENTS_DB:
        raise ValueError(f"Document with ID '{file_id}' does not exist in the system")
    
    # Check authorization
    # User can modify if they are the owner or have 'editor' role
    if account_id == owner_id or account_role == 'editor':
        # Update the content
        DOCUMENTS_DB[file_id]['content'] = updated_content
        return updated_content
    else:
        return "Error: You are not authorized to modify this document"
