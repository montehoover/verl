import logging

# Basic logging configuration for demonstration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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

def _is_authorized(account_id: str, account_role: str, document_owner_id: str) -> bool:
    """
    Checks if the user is authorized to modify the document.

    A user is authorized if they are the owner of the document or if they have an 'editor' role.

    Args:
        account_id: The ID of the user attempting the action.
        account_role: The role of the user.
        document_owner_id: The ID of the document's owner.

    Returns:
        True if authorized, False otherwise.
    """
    is_owner = account_id == document_owner_id
    is_editor = account_role == "editor"
    return is_owner or is_editor

def _create_updated_document_version(document: dict, new_content: str) -> dict:
    """
    Creates a new version of the document dictionary with updated content.

    This function is pure and returns a new dictionary, leaving the original unchanged.

    Args:
        document: The original document dictionary.
        new_content: The new content for the document.

    Returns:
        A new document dictionary with the content updated.
    """
    updated_document = document.copy()  # Create a shallow copy
    updated_document["content"] = new_content
    return updated_document

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

    if _is_authorized(account_id, account_role, document["owner_id"]):
        # Create a new version of the document with the updated content
        updated_document_data = _create_updated_document_version(document, updated_content)
        # Update the database with the new document data
        DOCUMENTS_DB[file_id] = updated_document_data
        logging.info(f"User '{account_id}' successfully modified document '{file_id}'.")
        return updated_document_data["content"]
    else:
        logging.warning(f"User '{account_id}' (role: '{account_role}') attempted to modify document '{file_id}' without authorization.")
        return "Error: User not authorized to modify this document."
