from typing import Dict, Any, List

DOCUMENTS_DB: Dict[str, Dict[str, Any]] = {
    "doc123": {
        "document_id": "doc123",
        "owner_id": "user111",
        "content": "Original content of doc123",
        "history": [
            {"version": 1, "content": "Initial draft of doc123"},
            {"version": 2, "content": "Revised intro in doc123"},
        ],
    },
    "doc456": {
        "document_id": "doc456",
        "owner_id": "user222",
        "content": "Original content of doc456",
        "history": [
            {"version": 1, "content": "Initial content of doc456"},
            {"version": 2, "content": "Minor typo fixes in doc456"},
        ],
    }
}


def modify_file(
    account_id: str,
    account_role: str,
    file_id: str,
    owner_id: str,
    updated_content: str
) -> Dict[str, Any]:
    """
    Modify the content of a specific document if the caller is authorized.

    Authorization:
      - Allowed if the user's role is 'editor' OR the user is the document's owner.
        Ownership is verified against the document's stored owner_id.

    Args:
        account_id: The user's unique ID.
        account_role: The user's role (e.g., 'viewer' or 'editor').
        file_id: The ID of the document to modify.
        owner_id: The expected owner ID for the document (validated against DB).
        updated_content: The new content to set for the document.

    Returns:
        A dict containing the document_id and the updated content.

    Raises:
        ValueError: If the document ID does not exist.
        PermissionError: If the caller is not authorized to modify the document.
    """
    if file_id not in DOCUMENTS_DB:
        raise ValueError(f"Document with id '{file_id}' does not exist")

    role = (account_role or "").strip().lower()
    is_editor = role == "editor"

    doc = DOCUMENTS_DB[file_id]
    doc_owner_id = doc.get("owner_id")

    # Consider the caller as owner if either the provided owner_id matches the DB
    # or the account_id matches the stored owner_id.
    is_owner = (owner_id == doc_owner_id) or (account_id == doc_owner_id)

    if not (is_editor or is_owner):
        raise PermissionError("You are not authorized to modify this document")

    # Optionally record history before modification (if history is tracked).
    # Ensure history list exists.
    history: List[Dict[str, Any]] = doc.setdefault("history", [])
    prev_content = doc.get("content", "")
    prev_version = history[-1]["version"] if history and "version" in history[-1] else 0
    history.append({"version": prev_version + 1, "content": prev_content})

    # Apply the update.
    doc["content"] = updated_content

    return {
        "document_id": doc["document_id"],
        "content": doc["content"],
    }
