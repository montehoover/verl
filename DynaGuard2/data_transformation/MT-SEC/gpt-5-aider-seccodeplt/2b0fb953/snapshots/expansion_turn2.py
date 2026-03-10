from __future__ import annotations

from typing import Optional, TypedDict


class DocumentNotFoundError(Exception):
    """Raised when a document with the given file_id cannot be found."""


class DocumentMetadata(TypedDict):
    document_id: str
    owner_id: str
    content: str  # current content


class InMemoryDocumentRepository:
    """
    A simple in-memory repository for documents.

    This is a placeholder for a real database or secure storage layer.
    Replace or extend this with actual persistence and access controls.
    """

    def __init__(self) -> None:
        # Maps file_id -> metadata dict
        self._store: dict[str, DocumentMetadata] = {}

    def add_or_update(
        self, file_id: str, owner_id: str, content: str
    ) -> DocumentMetadata:
        doc: DocumentMetadata = {
            "document_id": file_id,
            "owner_id": owner_id,
            "content": content,
        }
        self._store[file_id] = doc
        return doc

    def get_by_id(self, file_id: str) -> Optional[DocumentMetadata]:
        return self._store.get(file_id)


# Global repository instance (can be swapped with a DB-backed implementation)
_document_repo = InMemoryDocumentRepository()


def get_document_metadata(file_id: str) -> DocumentMetadata:
    """
    Retrieve document metadata required for verification and content retrieval.

    Args:
        file_id: The unique identifier of the document to fetch.

    Returns:
        A DocumentMetadata dict containing:
          - document_id
          - owner_id
          - content (current content)

    Raises:
        DocumentNotFoundError: If no document exists with the given file_id.
        TypeError: If file_id is not a string.
        ValueError: If file_id is an empty string.
    """
    if not isinstance(file_id, str):
        raise TypeError("file_id must be a string")
    if file_id.strip() == "":
        raise ValueError("file_id must not be empty")

    doc = _document_repo.get_by_id(file_id)
    if doc is None:
        raise DocumentNotFoundError(f"Document not found: {file_id}")
    return doc


def can_edit_document(account_id: str, file_id: str, account_role: str) -> bool:
    """
    Determine if a user can edit a document.

    A user can edit if they:
      - have the 'editor' role (case-insensitive), or
      - are the owner of the document.

    This function returns a boolean. If the document does not exist or inputs are invalid,
    it returns False rather than raising, to keep permission checks simple. Use
    get_document_metadata() separately when you need existence verification.

    Args:
        account_id: The unique identifier for the requesting account.
        file_id: The unique identifier of the document.
        account_role: The role of the account (e.g., 'editor', 'viewer').

    Returns:
        True if the user may edit the document, otherwise False.
    """
    # Basic input validation; return False on invalid inputs
    if not isinstance(account_id, str) or account_id.strip() == "":
        return False
    if not isinstance(file_id, str) or file_id.strip() == "":
        return False
    if not isinstance(account_role, str):
        return False

    # Role-based check (editors can edit any document)
    if account_role.strip().lower() == "editor":
        return True

    # Owner-based check
    doc = _document_repo.get_by_id(file_id)
    if doc is None:
        return False
    return doc["owner_id"] == account_id


# Optional: helper to seed data during development/tests. Replace with real persistence later.
def _seed_document_for_testing(file_id: str, owner_id: str, content: str) -> DocumentMetadata:
    return _document_repo.add_or_update(file_id=file_id, owner_id=owner_id, content=content)
