# Placeholder for document storage
# In a real system, this would be a database or a more robust storage solution.
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
    if file_id in DOCUMENTS_DB:
        doc_info = DOCUMENTS_DB[file_id]
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

def modify_file(account_id: str, account_role: str, file_id: str, owner_id: str, updated_content: str) -> str:
    """
    Modifies the content of a document if the user is authorized.

    Args:
        account_id: The ID of the account attempting the modification.
        account_role: The role of the account.
        file_id: The ID of the file (document) to modify.
        owner_id: The expected owner_id of the document.
        updated_content: The new content for the document.

    Returns:
        The updated content if successful.

    Raises:
        ValueError: If file_id does not exist or if provided owner_id does not match actual.
        PermissionError: If the user is not authorized to edit.
    """
    if file_id not in DOCUMENTS_DB:
        raise ValueError(f"Document ID '{file_id}' not found.")

    doc_data = DOCUMENTS_DB[file_id]
    actual_owner_id = doc_data["owner_id"]

    if owner_id != actual_owner_id:
        raise ValueError(f"Provided owner_id '{owner_id}' does not match the actual owner_id '{actual_owner_id}' for document '{file_id}'.")

    is_editor = (account_role == "editor")
    is_actual_owner = (account_id == actual_owner_id) # or account_id == owner_id, since they've been verified to be same

    if not (is_editor or is_actual_owner):
        raise PermissionError(f"User '{account_id}' not authorized to edit document '{file_id}'.")

    DOCUMENTS_DB[file_id]["content"] = updated_content
    return updated_content

if __name__ == '__main__':
    # Example usage for get_document_metadata:
    print("--- Example usage for get_document_metadata ---")
    metadata1 = get_document_metadata("doc123")
    if metadata1:
        print(f"Metadata for doc123: {metadata1}")
    else:
        print("File doc123 not found.")

    metadata2 = get_document_metadata("doc789") # Non-existent
    if metadata2:
        print(f"Metadata for doc789: {metadata2}")
    else:
        print("File doc789 not found.")

    print("\n--- Example usage for can_edit_document ---")
    # Test cases for can_edit_document (using new DB structure):
    # Owner trying to edit
    print(f"Can user111 (owner) edit doc123? {can_edit_document('user111', 'doc123', 'viewer')}") # True

    # Editor trying to edit
    print(f"Can user_editor (editor) edit doc123? {can_edit_document('user_editor', 'doc123', 'editor')}") # True

    # Non-owner, non-editor trying to edit
    print(f"Can user_viewer (viewer) edit doc123? {can_edit_document('user_viewer', 'doc123', 'viewer')}") # False

    # Trying to edit a non-existent file
    print(f"Can user111 edit non_existent_doc? {can_edit_document('user111', 'non_existent_doc', 'editor')}") # False

    # Owner of another document trying to edit doc123
    print(f"Can user222 (owner of doc456) edit doc123? {can_edit_document('user222', 'doc123', 'viewer')}") # False

    print("\n--- Example usage for modify_file ---")
    # Test cases for modify_file:
    # 1. Successful edit by owner
    try:
        print(f"Initial content of doc123: {DOCUMENTS_DB['doc123']['content']}")
        updated_content = modify_file(account_id="user111", account_role="viewer", file_id="doc123", owner_id="user111", updated_content="New content by owner user111.")
        print(f"doc123 updated by owner. New content: '{updated_content}'")
        print(f"Content in DB after update: {DOCUMENTS_DB['doc123']['content']}")
    except Exception as e:
        print(f"Error modifying doc123 by owner: {e}")

    # 2. Successful edit by editor
    try:
        print(f"Initial content of doc456: {DOCUMENTS_DB['doc456']['content']}")
        updated_content_editor = modify_file(account_id="editor_user", account_role="editor", file_id="doc456", owner_id="user222", updated_content="New content by editor_user.")
        print(f"doc456 updated by editor. New content: '{updated_content_editor}'")
        print(f"Content in DB after update: {DOCUMENTS_DB['doc456']['content']}")
    except Exception as e:
        print(f"Error modifying doc456 by editor: {e}")

    # 3. Unauthorized edit attempt (not owner, not editor)
    try:
        modify_file(account_id="random_user", account_role="viewer", file_id="doc123", owner_id="user111", updated_content="Attempt by random_user.")
        print("doc123 modification by random_user succeeded (unexpected).")
    except PermissionError as e:
        print(f"Caught expected error for random_user on doc123: {e}")
    except Exception as e:
        print(f"Caught unexpected error for random_user on doc123: {e}")

    # 4. Attempt to edit non-existent document
    try:
        modify_file(account_id="user111", account_role="editor", file_id="doc789", owner_id="user_unknown", updated_content="Attempt on non-existent doc.")
        print("doc789 modification succeeded (unexpected).")
    except ValueError as e:
        print(f"Caught expected error for non-existent doc789: {e}")
    except Exception as e:
        print(f"Caught unexpected error for non-existent doc789: {e}")

    # 5. Attempt with incorrect owner_id provided
    try:
        # user111 is owner of doc123, but we provide "wrong_owner"
        modify_file(account_id="user111", account_role="viewer", file_id="doc123", owner_id="wrong_owner", updated_content="Attempt with wrong owner_id.")
        print("doc123 modification with wrong owner_id succeeded (unexpected).")
    except ValueError as e:
        print(f"Caught expected error for wrong owner_id on doc123: {e}")
    except Exception as e:
        print(f"Caught unexpected error for wrong owner_id on doc123: {e}")

    # 6. Unauthorized: Correct owner_id provided, but account_id is not owner and not editor
    try:
        # doc456 is owned by user222. editor_user is an editor. some_other_user is neither.
        modify_file(account_id="some_other_user", account_role="viewer", file_id="doc456", owner_id="user222", updated_content="Attempt by some_other_user on doc456.")
        print("doc456 modification by some_other_user succeeded (unexpected).")
    except PermissionError as e:
        print(f"Caught expected error for some_other_user on doc456: {e}")
    except Exception as e:
        print(f"Caught unexpected error for some_other_user on doc456: {e}")
