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

def modify_file(account_id: str, account_role: str, file_id: str) -> str:
    """
    Retrieves the content of a specific document based on the user's role.

    Args:
        account_id: The user's unique ID.
        account_role: The user's role (e.g., 'viewer', 'editor').
        file_id: The ID of the document to be accessed.

    Returns:
        The content of the document.

    Raises:
        ValueError: If the document ID doesn't exist in the system.
        PermissionError: If the account role is not authorized to view the document.
    """
    if file_id not in DOCUMENTS_DB:
        raise ValueError(f"Document ID '{file_id}' does not exist.")

    document = DOCUMENTS_DB[file_id]

    if account_role == 'editor':
        # Editors can view the document content
        # Editors have additional privileges (placeholder, not implemented here)
        print(f"User {account_id} (editor) has additional privileges for document {file_id} (not yet implemented).")
        return document["content"]
    elif account_role == 'viewer':
        # Viewers can view the document content
        return document["content"]
    else:
        raise PermissionError(f"Account role '{account_role}' is not authorized to access document '{file_id}'.")

if __name__ == '__main__':
    # Example Usage:
    try:
        # Viewer access
        content_viewer = modify_file("user001", "viewer", "doc123")
        print(f"Viewer access to doc123: {content_viewer}")

        # Editor access
        content_editor = modify_file("user002", "editor", "doc456")
        print(f"Editor access to doc456: {content_editor}")

        # Attempt access with an invalid document ID
        # modify_file("user003", "viewer", "doc789") 
    except ValueError as ve:
        print(f"Error: {ve}")
    except PermissionError as pe:
        print(f"Error: {pe}")

    try:
        # Attempt access with an invalid role
        modify_file("user004", "guest", "doc123")
    except ValueError as ve:
        print(f"Error: {ve}")
    except PermissionError as pe:
        print(f"Error: {pe}")
    
    try:
        # Attempt access to non-existent document
        modify_file("user005", "viewer", "doc999")
    except ValueError as ve:
        print(f"Error: {ve}")
    except PermissionError as pe:
        print(f"Error: {pe}")
