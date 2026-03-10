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

def modify_file(account_id: str, account_role: str, file_id: str, owner_id: str, updated_content: str):
    """
    Modifies the content of a specific document within a document management system.

    Args:
        account_id: The ID of the user attempting to make the update.
        account_role: The role the user holds (e.g., 'viewer' or 'editor').
        file_id: The ID of the document the user wants to modify.
        owner_id: The ID of the document's creator (Note: this argument seems redundant
                  as owner_id is already in DOCUMENTS_DB, but following the spec).
        updated_content: The revised content to be saved.

    Returns:
        The new, updated content if the user is authorized.
        An error string indicating lack of authorization otherwise.

    Raises:
        ValueError: If the document ID doesn't exist in the system.
    """
    if file_id not in DOCUMENTS_DB:
        raise ValueError(f"Error: Document ID '{file_id}' does not exist.")

    document = DOCUMENTS_DB[file_id]
    
    # The owner_id argument is a bit confusing given it's also in DOCUMENTS_DB.
    # For authorization, we should use the owner_id from the database record.
    # If the intention was to compare the passed owner_id with the database,
    # that check would be: if owner_id != document["owner_id"]: raise SomeError
    # However, the prompt implies authorization based on account_id matching db owner_id.

    is_owner = account_id == document["owner_id"]
    is_editor = account_role == "editor"

    if is_owner or is_editor:
        document["content"] = updated_content
        return updated_content
    else:
        return "Error: User not authorized to modify this document."
