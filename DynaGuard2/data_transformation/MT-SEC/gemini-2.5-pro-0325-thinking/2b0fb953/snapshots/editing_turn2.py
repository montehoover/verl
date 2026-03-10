DOCUMENTS_DB = {
    "doc123": {
        "document_id": "doc123",
        "owner_id": "user111",
        "content": "Original content of doc123",
        "history": ["Initial version of doc123"]
    },
    "doc456": {
        "document_id": "doc456",
        "owner_id": "user222",
        "content": "Original content of doc456",
        "history": ["Initial version of doc456", "Updated content for doc456"]
    }
}

def modify_file(account_id: str, account_role: str, file_id: str) -> tuple[str, list[str]]:
    """
    Retrieves the content and history of a specific document based on the user's role.

    Args:
        account_id: The user's unique ID.
        account_role: The user's role (e.g., 'viewer', 'editor').
        file_id: The ID of the document to be accessed.

    Returns:
        A tuple containing the document content (str) and its history (list[str]).

    Raises:
        ValueError: If the document ID doesn't exist in the system.
        PermissionError: If the account role is not authorized to view the document.
    """
    if file_id not in DOCUMENTS_DB:
        raise ValueError(f"Document ID '{file_id}' does not exist.")

    document = DOCUMENTS_DB[file_id]

    document_content = document["content"]
    document_history = document.get("history", []) # Default to empty list if no history

    if account_role == 'editor':
        # Editors can view the document content and history
        # Editors have additional privileges (placeholder, not implemented here)
        print(f"User {account_id} (editor) has additional privileges for document {file_id} (not yet implemented).")
        return document_content, document_history
    elif account_role == 'viewer':
        # Viewers can view the document content and history
        return document_content, document_history
    else:
        raise PermissionError(f"Account role '{account_role}' is not authorized to access document '{file_id}'.")

if __name__ == '__main__':
    # Example Usage:
    try:
        # Viewer access
        content_viewer, history_viewer = modify_file("user001", "viewer", "doc123")
        print(f"Viewer access to doc123 content: {content_viewer}")
        print(f"Viewer access to doc123 history: {history_viewer}")

        # Editor access
        content_editor, history_editor = modify_file("user002", "editor", "doc456")
        print(f"Editor access to doc456 content: {content_editor}")
        print(f"Editor access to doc456 history: {history_editor}")

        # Attempt access with an invalid document ID
        # content_invalid, history_invalid = modify_file("user003", "viewer", "doc789")
    except ValueError as ve:
        print(f"Error: {ve}")
    except PermissionError as pe:
        print(f"Error: {pe}")

    try:
        # Attempt access with an invalid role
        content_guest, history_guest = modify_file("user004", "guest", "doc123")
    except ValueError as ve:
        print(f"Error: {ve}")
    except PermissionError as pe:
        print(f"Error: {pe}")
    
    try:
        # Attempt access to non-existent document
        content_nonexistent, history_nonexistent = modify_file("user005", "viewer", "doc999")
    except ValueError as ve:
        print(f"Error: {ve}")
    except PermissionError as pe:
        print(f"Error: {pe}")
