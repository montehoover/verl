import logging
import time
from datetime import datetime, timezone

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


def _get_audit_logger() -> logging.Logger:
    """
    Configure and return a dedicated audit logger.
    Ensures UTC timestamps and a clear, human-readable format.
    """
    logger = logging.getLogger("document_audit")
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)s | %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%SZ",
        )
        # Force UTC time in asctime
        formatter.converter = time.gmtime
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger


def _log_document_change(operator_id: str, doc_id: str, action: str = "document_altered") -> None:
    """
    Write a structured audit log for document changes.

    Args:
        operator_id: The ID of the user performing the action.
        doc_id: The unique ID of the affected document.
        action: The action performed (default: 'document_altered').
    """
    logger = _get_audit_logger()
    iso_ts = datetime.now(timezone.utc).isoformat()
    # Key=value format for easy parsing and readability
    logger.info(
        "action=%s operator_id=%s doc_id=%s timestamp=%s",
        action,
        operator_id,
        doc_id,
        iso_ts,
    )


def _check_permissions(
    operator_id: str,
    operator_role: str,
    provided_owner_id: str,
    actual_owner_id: str
) -> None:
    """
    Pure function to validate permissions for modifying a document.

    Raises:
        PermissionError: If any permission rule is violated.
    """
    # Validate the owner matches the database record
    if provided_owner_id != actual_owner_id:
        raise PermissionError("Insufficient permissions: document owner mismatch.")

    # Validate operator privileges
    if operator_role != "editor":
        raise PermissionError("Insufficient permissions: only editors can modify documents.")

    # Only the owner is allowed to modify the document
    if operator_id != provided_owner_id:
        raise PermissionError("Insufficient permissions: only the document owner can modify this document.")


def _build_updated_document(document: dict, updated_content: str) -> dict:
    """
    Pure function to create an updated document structure without side effects.

    Args:
        document: The current document dict.
        updated_content: The new content to set.

    Returns:
        A new document dict with updated content.
    """
    return {
        "document_id": document.get("document_id"),
        "owner_id": document.get("owner_id"),
        "content": updated_content,
    }


def alter_document(
    operator_id: str,
    operator_role: str,
    doc_id: str,
    doc_owner_id: str,
    updated_content: str
) -> str:
    """
    Modify the content of an existing document if the operator is authorized.

    Authorization rules:
    - The document must exist.
    - The provided doc_owner_id must match the document's actual owner.
    - The operator must have role 'editor'.
    - Only the owner (operator_id == doc_owner_id) is allowed to modify the document.

    Returns:
        str: The updated content of the document.

    Raises:
        KeyError: If the document does not exist.
        PermissionError: If the operator is not authorized to modify the document.
    """
    # Ensure the document exists
    if doc_id not in DOCUMENTS_DB:
        raise KeyError(f"Document not found: {doc_id}")

    document = DOCUMENTS_DB[doc_id]

    # Permission checks (pure)
    _check_permissions(
        operator_id=operator_id,
        operator_role=operator_role,
        provided_owner_id=doc_owner_id,
        actual_owner_id=document.get("owner_id"),
    )

    # Build an updated document (pure) and persist the change
    updated_document = _build_updated_document(document, updated_content)
    DOCUMENTS_DB[doc_id] = updated_document

    # Audit log for successful change
    _log_document_change(operator_id=operator_id, doc_id=doc_id, action="document_altered")

    return updated_document["content"]
