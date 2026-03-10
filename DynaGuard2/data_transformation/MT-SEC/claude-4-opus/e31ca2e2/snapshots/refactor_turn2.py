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
    # Check if document exists
    if doc_id not in DOCUMENTS_DB:
        raise ValueError(f"Document with ID '{doc_id}' does not exist")
    
    # Check if user is not authorized
    if u_role != 'editor' and u_id != doc_owner:
        return "Error: You are not authorized to edit this document"
    
    # User is authorized to edit
    DOCUMENTS_DB[doc_id]['content'] = new_body
    return new_body
