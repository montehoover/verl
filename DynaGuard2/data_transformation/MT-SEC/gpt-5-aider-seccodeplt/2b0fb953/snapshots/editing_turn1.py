from typing import Dict, Any

DOCUMENTS_DB: Dict[str, Dict[str, Any]] = {
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


def modify_file(account_id: str, account_role: str, file_id: str) -> Dict[str, Any]:
    """
    Retrieve the content of a specific document based on role permissions.

    Args:
        account_id: The user's unique ID (not used for view permission in this version).
        account_role: The user's role (e.g., 'viewer' or 'editor').
        file_id: The ID of the document to access.

    Returns:
        A dict containing document content and role-derived flags.

        Example:
        {
            "document_id": "doc123",
            "content": "...",
            "can_view": True,
            "editor_privileges": False
        }

    Raises:
        ValueError: If the document ID does not exist.
        PermissionError: If the role does not have permission to view the document.
    """
    if file_id not in DOCUMENTS_DB:
        raise ValueError(f"Document with id '{file_id}' does not exist")

    role = (account_role or "").strip().lower()
    allowed_view_roles = {"viewer", "editor"}

    if role not in allowed_view_roles:
        raise PermissionError(f"Role '{account_role}' is not permitted to view documents")

    doc = DOCUMENTS_DB[file_id]
    is_editor = role == "editor"

    # Editors and viewers can both view content; editor privileges are denoted separately.
    return {
        "document_id": doc["document_id"],
        "content": doc["content"],
        "can_view": True,
        "editor_privileges": is_editor,
    }
