DOCUMENTS_DB = {
    "doc123": {
        "document_id": "doc123",
        "owner_id": "user111",
        "content": "Original content of doc123",
        "history": [
            {"version": 1, "content": "Initial content of doc123", "timestamp": "2024-01-01T10:00:00"},
            {"version": 2, "content": "Original content of doc123", "timestamp": "2024-01-02T14:30:00"}
        ]
    },
    "doc456": {
        "document_id": "doc456",
        "owner_id": "user222",
        "content": "Original content of doc456",
        "history": [
            {"version": 1, "content": "Initial content of doc456", "timestamp": "2024-01-01T11:00:00"},
            {"version": 2, "content": "Original content of doc456", "timestamp": "2024-01-03T09:15:00"}
        ]
    }
}

def modify_file(account_id, account_role, file_id):
    # Check if the document exists
    if file_id not in DOCUMENTS_DB:
        raise ValueError(f"Document ID '{file_id}' doesn't exist in the system")
    
    # Check permissions - both viewers and editors can view
    if account_role in ['viewer', 'editor']:
        document = DOCUMENTS_DB[file_id]
        result = {
            'content': document['content'],
            'history': document.get('history', [])
        }
        return result
    else:
        raise ValueError(f"Invalid role '{account_role}'. Must be 'viewer' or 'editor'")
