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
        raise ValueError(f"Document ID '{doc_id}' doesn't exist in the system")
    
    # Check authorization: user must be either the owner or have 'editor' role
    if u_id == doc_owner or u_role == 'editor':
        # User is authorized, update the document
        DOCUMENTS_DB[doc_id]['content'] = new_body
        return new_body
    else:
        # User is not authorized
        return "Error: You are not authorized to edit this document"
