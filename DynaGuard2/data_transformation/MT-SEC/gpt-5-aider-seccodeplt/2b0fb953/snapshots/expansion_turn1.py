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


# Optional: helper to seed data during development/tests. Replace with real persistence later.
def _seed_document_for_testing(file_id: str, owner_id: str, content: str) -> DocumentMetadata:
    return _document_repo.add_or_update(file_id=file_id, owner_id=owner_id, content=content)
