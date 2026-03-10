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

def edit_doc(u_id, u_role, doc_id, doc_owner, new_body):
    if doc_id not in DOCUMENTS_DB:
        raise ValueError(f"Document ID {doc_id} doesn't exist")
    
    if u_role == 'editor' or u_id == DOCUMENTS_DB[doc_id]['owner_id']:
        DOCUMENTS_DB[doc_id]['content'] = new_body
        return new_body
    else:
        raise ValueError(f"Unauthorized access for user {u_id} with role {u_role}")
