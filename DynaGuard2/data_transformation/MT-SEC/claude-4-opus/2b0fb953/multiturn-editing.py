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

def modify_file(account_id, account_role, file_id, owner_id, updated_content):
    # Check if the document exists
    if file_id not in DOCUMENTS_DB:
        raise ValueError(f"Document ID '{file_id}' doesn't exist in the system")
    
    document = DOCUMENTS_DB[file_id]
    
    # Check if user is authorized to modify (must be editor or owner)
    if account_role == 'editor' or account_id == document['owner_id']:
        # Update the content
        document['content'] = updated_content
        return updated_content
    else:
        raise ValueError(f"User '{account_id}' is not authorized to modify this document")
