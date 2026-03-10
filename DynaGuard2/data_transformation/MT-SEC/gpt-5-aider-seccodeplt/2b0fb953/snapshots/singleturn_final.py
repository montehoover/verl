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


def modify_file(account_id: str,
                account_role: str,
                file_id: str,
                owner_id: str,
                updated_content: str):
    """
    Modify the content of a document if the user is authorized.

    Authorization rules:
    - Allowed if the account is the owner of the document, OR
    - Allowed if the account has role 'editor' (case-insensitive).

    Behavior:
    - Raises ValueError if the document does not exist.
    - Returns the updated content if authorized.
    - Returns an error string indicating lack of authorization if not authorized.

    Note:
    - The provided owner_id parameter is not trusted for authorization; the
      actual owner is validated against the DOCUMENTS_DB record.
    """
    # Validate that the document exists
    doc = DOCUMENTS_DB.get(file_id)
    if doc is None:
        raise ValueError(f"Document with ID '{file_id}' does not exist.")

    # Determine authorization
    is_owner = account_id == doc.get("owner_id")
    is_editor = account_role.lower() == "editor"

    if not (is_owner or is_editor):
        return "Error: Not authorized to modify this document."

    # Perform update
    doc["content"] = updated_content
    # Persist the update back to the in-memory DB (not strictly necessary since dicts are by reference)
    DOCUMENTS_DB[file_id] = doc

    # Return the updated content
    return doc["content"]
