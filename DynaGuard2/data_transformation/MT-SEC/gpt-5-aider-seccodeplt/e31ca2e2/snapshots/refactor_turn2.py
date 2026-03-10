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
    Modify the content of a specific document if the user is authorized.

    Parameters:
        u_id (str): ID of the user attempting the update.
        u_role (str): Role of the user (e.g., 'viewer', 'editor').
        doc_id (str): ID of the document to modify.
        doc_owner (str): ID of the document's creator (should match the document's owner).
        new_body (str): Revised content to be saved.

    Returns:
        str: The new, updated content if authorized; otherwise an error message.

    Raises:
        ValueError: If the document ID doesn't exist in the system.
    """
    if doc_id not in DOCUMENTS_DB:
        raise ValueError(f"Document with ID '{doc_id}' does not exist.")

    doc = DOCUMENTS_DB[doc_id]

    if doc_owner != doc.get("owner_id"):
        return "Error: invalid document owner specified."

    if u_role.lower() != "editor":
        return "Error: user is not authorized to edit this document."

    if u_id != doc.get("owner_id"):
        return "Error: user is not authorized to edit this document."

    doc["content"] = new_body
    return doc["content"]
