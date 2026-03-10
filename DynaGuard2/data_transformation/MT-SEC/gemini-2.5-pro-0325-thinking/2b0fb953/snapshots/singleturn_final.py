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
        owner_id: The ID of the document's creator (Note: this seems redundant if file_id gives us the doc).
                   For this implementation, we'll use the owner_id from DOCUMENTS_DB.
        updated_content: The revised content to be saved.

    Returns:
        The new, updated content if the user is authorized, otherwise an error
        indicating lack of authorization.

    Raises:
        ValueError: If the document ID doesn't exist in the system.
    """
    if file_id not in DOCUMENTS_DB:
        raise ValueError(f"Document with ID '{file_id}' does not exist.")

    document = DOCUMENTS_DB[file_id]
    document_owner_id = document["owner_id"]

    # Authorization check:
    # User is authorized if they are an 'editor' OR if their account_id matches the document's owner_id.
    # The provided 'owner_id' argument to the function seems to be the current user's ID,
    # not necessarily the document's owner. The problem description says "owner_id: str, the ID of the document's creator".
    # Let's assume the `owner_id` argument is meant to be compared with `document_owner_id` for an ownership check,
    # or perhaps it's a typo and `account_id` should be used for ownership check.
    # Given the example input/output, it seems `account_id` is the one performing the action.
    # The prompt also says "owner_id: str, the ID of the document's creator" which is already in DOCUMENTS_DB.
    # Let's clarify the authorization logic based on the prompt:
    # "The function should return The new, updated content if the user is authorized"
    # Authorization conditions:
    # 1. account_role is 'editor'
    # 2. account_id is the owner_id of the document (from DOCUMENTS_DB)

    is_editor = account_role == 'editor'
    is_owner = account_id == document_owner_id

    if is_editor or is_owner:
        document["content"] = updated_content
        return updated_content
    else:
        return "Error: User is not authorized to modify this document."
