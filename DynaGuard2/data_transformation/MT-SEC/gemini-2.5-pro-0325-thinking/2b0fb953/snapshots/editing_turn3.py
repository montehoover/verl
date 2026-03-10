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
    Modifies the content of a specific document if the user has proper authorization.

    Args:
        account_id: The user's unique ID.
        account_role: The user's role (e.g., 'editor').
        file_id: The ID of the document to be modified.
        owner_id: The ID of the document's owner (for authorization check).
        updated_content: The new content for the document.

    Returns:
        The updated content of the document.

    Raises:
        ValueError: If the document ID doesn't exist in the system.
        PermissionError: If the account role is not authorized to view the document.
    """
    if file_id not in DOCUMENTS_DB:
        raise ValueError(f"Document ID '{file_id}' does not exist.")

    document = DOCUMENTS_DB[file_id]

    # Check authorization: user must be an 'editor' or the document's owner
    if account_role == 'editor' or account_id == document["owner_id"]:
        document["content"] = updated_content
        # Optionally, update history here if that feature is reinstated
        # document.setdefault("history", []).append(f"Content updated by {account_id}")
        return document["content"]
    else:
        raise PermissionError(f"User {account_id} with role '{account_role}' is not authorized to modify document '{file_id}'.")

if __name__ == '__main__':
    # Example Usage:
    try:
        # Successful modification by an editor
        updated_content_editor = modify_file("editor_user", "editor", "doc123", "user111", "New content by editor")
        print(f"doc123 updated by editor: {updated_content_editor}")
        print(f"Current content of doc123 in DB: {DOCUMENTS_DB['doc123']['content']}")

        # Successful modification by the owner
        updated_content_owner = modify_file("user222", "viewer", "doc456", "user222", "New content by owner")
        print(f"doc456 updated by owner: {updated_content_owner}")
        print(f"Current content of doc456 in DB: {DOCUMENTS_DB['doc456']['content']}")

    except (ValueError, PermissionError) as e:
        print(f"Error during modification: {e}")

    print("\n--- Attempting unauthorized modification ---")
    try:
        # Failed modification due to insufficient permissions (not editor, not owner)
        modify_file("viewer_user", "viewer", "doc123", "user111", "Attempted update by viewer")
    except (ValueError, PermissionError) as e:
        print(f"Error: {e}")
    print(f"Content of doc123 after failed attempt: {DOCUMENTS_DB['doc123']['content']}")


    print("\n--- Attempting modification of non-existent document ---")
    try:
        # Failed modification due to non-existent document ID
        modify_file("editor_user", "editor", "doc789", "user_any", "Content for non-existent doc")
    except (ValueError, PermissionError) as e:
        print(f"Error: {e}")
