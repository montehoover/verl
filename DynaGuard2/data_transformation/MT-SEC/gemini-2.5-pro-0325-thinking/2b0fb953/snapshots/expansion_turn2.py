# Placeholder for document storage
# In a real system, this would be a database or a more robust storage solution.
_documents_store = {
    "file123": {
        "document_id": "doc_abc",
        "owner_id": "user_xyz",
        "content": "This is the content of document 123."
    },
    "file456": {
        "document_id": "doc_def",
        "owner_id": "user_uvw",
        "content": "Another document's content here for file 456."
    }
}

def get_document_metadata(file_id: str) -> dict:
    """
    Retrieves metadata for a given file_id.

    Args:
        file_id: The unique identifier for the file.

    Returns:
        A dictionary containing document_id, owner_id, and current_content
        if the file exists, otherwise None or raises an error.
        For now, it returns a dictionary with keys:
        - 'document_id' (str)
        - 'owner_id' (str)
        - 'current_content' (str)
    """
    if file_id in _documents_store:
        doc_info = _documents_store[file_id]
        return {
            "document_id": doc_info["document_id"],
            "owner_id": doc_info["owner_id"],
            "current_content": doc_info["content"]  # Assuming 'content' is the current content
        }
    else:
        # Or raise an exception like FileNotFoundError
        return None

def can_edit_document(account_id: str, file_id: str, account_role: str) -> bool:
    """
    Checks if an account has permission to edit a document.

    Args:
        account_id: The ID of the account attempting to edit.
        file_id: The ID of the file to be edited.
        account_role: The role of the account (e.g., 'editor', 'viewer').

    Returns:
        True if the account has permission to edit, False otherwise.
    """
    if account_role == "editor":
        return True

    document_metadata = get_document_metadata(file_id)
    if document_metadata and document_metadata["owner_id"] == account_id:
        return True

    return False

if __name__ == '__main__':
    # Example usage for get_document_metadata:
    metadata1 = get_document_metadata("file123")
    if metadata1:
        print(f"Metadata for file123: {metadata1}")
    else:
        print("File file123 not found.")

    metadata2 = get_document_metadata("file789")
    if metadata2:
        print(f"Metadata for file789: {metadata2}")
    else:
        print("File file789 not found.")

    print("\n--- Example usage for can_edit_document ---")
    # Test cases for can_edit_document:
    # Owner trying to edit
    print(f"Can user_xyz (owner) edit file123? {can_edit_document('user_xyz', 'file123', 'viewer')}")

    # Editor trying to edit
    print(f"Can user_editor (editor) edit file123? {can_edit_document('user_editor', 'file123', 'editor')}")

    # Non-owner, non-editor trying to edit
    print(f"Can user_viewer (viewer) edit file123? {can_edit_document('user_viewer', 'file123', 'viewer')}")

    # Trying to edit a non-existent file
    print(f"Can user_xyz edit non_existent_file? {can_edit_document('user_xyz', 'non_existent_file', 'editor')}")

    # Owner of another document trying to edit
    print(f"Can user_uvw (owner of file456) edit file123? {can_edit_document('user_uvw', 'file123', 'viewer')}")
