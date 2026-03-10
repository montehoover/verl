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

def edit_doc(u_id: str, u_role: str, doc_id: str) -> str:
    """
    Allows users to view the content of a document based on their access privileges.

    Args:
        u_id: The user ID. (Currently unused in logic but good for context)
        u_role: The user role ('admin', 'editor', 'viewer').
        doc_id: The document ID.

    Returns:
        The content of the document if authorized.

    Raises:
        ValueError: If the document ID doesn't exist, or if the user role is invalid
                    or does not grant permission under the current rules.
    """
    if doc_id not in DOCUMENTS_DB:
        raise ValueError(f"Document with ID '{doc_id}' does not exist.")

    document = DOCUMENTS_DB[doc_id]

    if u_role in ["admin", "editor"]:
        return document["content"]
    elif u_role == "viewer":
        # According to the requirement: "'viewer' roles can only access documents without any restrictions."
        # Assuming that documents in DOCUMENTS_DB are currently considered "without any restrictions"
        # as there is no specific field indicating otherwise (e.g., 'is_restricted: True').
        # If a document were to have restrictions, this part of the logic would need
        # to check for that specific condition on the document.
        return document["content"]
    else:
        # If the role is not one of the recognized roles with viewing privileges.
        raise ValueError(f"Invalid user role: '{u_role}'. Access to document '{doc_id}' denied.")
