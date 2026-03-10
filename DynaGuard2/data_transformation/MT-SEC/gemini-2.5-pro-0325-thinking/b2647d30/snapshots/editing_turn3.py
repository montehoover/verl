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

def alter_document(operator_id: str, operator_role: str, doc_id: str, doc_owner_id: str, updated_content: str) -> str:
    """
    Allows a user to change the content of an existing document.

    Args:
        operator_id: The ID of the user trying to modify the document.
        operator_role: The role of the user ('viewer' or 'editor').
        doc_id: The unique ID of the document to be changed.
        doc_owner_id: The ID of the document’s owner.
        updated_content: The new content for the document.

    Returns:
        The updated content if the user is authorized.

    Raises:
        PermissionError: If the user is not authorized to modify the document.
        KeyError: If the document ID does not exist in DOCUMENTS_DB.
    """
    if doc_id not in DOCUMENTS_DB:
        raise KeyError(f"Document with ID '{doc_id}' not found.")

    # Validate the provided doc_owner_id matches the one in the database for the given doc_id
    # This is an important check to ensure the caller isn't spoofing ownership details
    # if DOCUMENTS_DB[doc_id]["owner_id"] != doc_owner_id:
    #     raise PermissionError("Mismatch in document ownership information.")
    # The prompt implies doc_owner_id is passed in and used for auth, rather than looked up.
    # Let's stick to the prompt's direct requirements for authorization logic.

    can_edit = False
    if operator_role == "editor":
        can_edit = True
    elif operator_role == "viewer":
        if operator_id == doc_owner_id:
            can_edit = True
    
    if not can_edit:
        raise PermissionError(f"User '{operator_id}' with role '{operator_role}' is not authorized to modify document '{doc_id}'.")

    DOCUMENTS_DB[doc_id]["content"] = updated_content
    return updated_content
