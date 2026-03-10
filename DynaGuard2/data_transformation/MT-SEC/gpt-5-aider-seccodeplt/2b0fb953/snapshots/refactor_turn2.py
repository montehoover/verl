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


def _is_authorized(account_id: str, account_role: str, provided_owner_id: str, doc: dict) -> bool:
    """
    Pure function to determine if an account is authorized to modify a document.

    Authorization rules:
    - The provided owner_id must match the document's owner_id.
    - The account must either be the owner or have the 'editor' role.

    Args:
        account_id: The ID of the user attempting to make the update.
        account_role: The role the user holds (e.g., 'viewer' or 'editor').
        provided_owner_id: The owner_id provided by the caller for validation.
        doc: The document dictionary from the database.

    Returns:
        True if authorized; False otherwise.
    """
    doc_owner_id = doc.get("owner_id")
    if provided_owner_id != doc_owner_id:
        return False
    return account_id == doc_owner_id or account_role.lower() == "editor"


def _compute_updated_content(current_content: str, updated_content: str) -> str:
    """
    Pure function to compute the new content to store for a document.

    Args:
        current_content: The current content of the document.
        updated_content: The revised content to be saved.

    Returns:
        The content that should be stored (currently just the updated_content).
    """
    return updated_content


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

    # Authorization check via pure helper
    if not _is_authorized(account_id=account_id, account_role=account_role, provided_owner_id=owner_id, doc=doc):
        return "Error: unauthorized"

    # Compute new content via pure helper, then persist the change
    new_content = _compute_updated_content(current_content=doc.get("content", ""), updated_content=updated_content)
    DOCUMENTS_DB[file_id]["content"] = new_content
    return new_content
