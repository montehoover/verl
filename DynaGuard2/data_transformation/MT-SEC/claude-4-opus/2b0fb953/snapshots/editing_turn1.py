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

def modify_file(account_id, account_role, file_id):
    # Check if the document exists
    if file_id not in DOCUMENTS_DB:
        raise ValueError(f"Document ID '{file_id}' doesn't exist in the system")
    
    # Check permissions - both viewers and editors can view
    if account_role in ['viewer', 'editor']:
        document = DOCUMENTS_DB[file_id]
        return document['content']
    else:
        raise ValueError(f"Invalid role '{account_role}'. Must be 'viewer' or 'editor'")
