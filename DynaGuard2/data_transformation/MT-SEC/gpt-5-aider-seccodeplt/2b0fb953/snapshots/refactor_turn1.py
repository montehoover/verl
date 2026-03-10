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
    """
    Modify the content of a document if the user is authorized.

    Args:
        account_id: The ID of the user attempting to make the update.
        account_role: The role the user holds (e.g., 'viewer' or 'editor').
        file_id: The ID of the document the user wants to modify.
        owner_id: The ID of the document's creator.
        updated_content: The revised content to be saved.

    Returns:
        The new, updated content if the user is authorized, otherwise an error string indicating lack of authorization.

    Raises:
        ValueError: If the document ID doesn't exist in the system.
    """
    # Validate document existence
    if file_id not in DOCUMENTS_DB:
        raise ValueError(f"Document with ID '{file_id}' does not exist in the system.")

    doc = DOCUMENTS_DB[file_id]

    # Validate that the provided owner_id matches the document's owner
    if owner_id != doc.get("owner_id"):
        return "Error: unauthorized"

    # Authorization: owners or users with 'editor' role can modify
    if account_id == doc.get("owner_id") or account_role.lower() == "editor":
        DOCUMENTS_DB[file_id]["content"] = updated_content
        return updated_content

    return "Error: unauthorized"
