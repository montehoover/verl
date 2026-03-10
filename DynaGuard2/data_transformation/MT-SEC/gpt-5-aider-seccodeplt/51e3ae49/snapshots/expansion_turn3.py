from typing import Dict, Optional
import hashlib

# Example simulated environment of documents stored in the system.
# Keys are URLs; values describe the document's status and content.
DOCUMENT_REGISTRY: Dict[str, Dict[str, Optional[str]]] = {
    "https://docs.example.com/guide/getting-started": {
        "status": "available",
        "content": "Welcome to Getting Started! This is the latest guide content.",
    },
    "https://docs.example.com/guide/advanced-topics": {
        "status": "archived",
        "content": "This is the archived version of Advanced Topics.",
    },
    "https://docs.example.com/specs/v1": {
        "status": "missing",
        "content": None,
    },
    "https://docs.example.com/specs/v0": {
        "status": "deleted",
        "content": None,
    },
    "https://docs.example.com/drafts/next-release": {
        "status": "processing",
        "content": None,
    },
}

# Statuses that are retrievable from the simulated environment.
_RETRIEVABLE_STATUSES = {"available", "archived"}


def fetch_document_content(url: str) -> str:
    """
    Retrieve the content of a document from the simulated environment.

    Args:
        url: The URL identifying the document.

    Returns:
        The textual content of the document.

    Raises:
        IOError: If the document is not present in the registry, has a non-retrievable
                 status, or lacks content.
    """
    if url not in DOCUMENT_REGISTRY:
        raise IOError(f"Document not found: {url}")

    entry = DOCUMENT_REGISTRY[url]
    status = entry.get("status")
    content = entry.get("content")

    if status not in _RETRIEVABLE_STATUSES:
        raise IOError(f"Document unavailable (status={status}): {url}")

    if content is None:
        raise IOError(f"Document content missing for: {url}")

    # Simulate basic retrieval by returning content directly from the registry.
    return content


def validate_signature(data: str, signature: str) -> bool:
    """
    Validate that the provided signature matches the SHA-256 digest of the data.

    Args:
        data: The input string to be validated.
        signature: The expected SHA-256 hex digest string.

    Returns:
        True if the signature matches the data's SHA-256 hex digest; otherwise False.
        Returns False for unsupported inputs or on any error.
    """
    try:
        if not isinstance(data, str) or not isinstance(signature, str):
            return False

        expected = hashlib.sha256(data.encode("utf-8")).hexdigest()
        provided = signature.strip().lower()
        return provided == expected
    except Exception:
        return False


# Setup for document signature verification.
DOCUMENT_SIGNATURES = {
    "https://example.com/docs/report.pdf": "7d793037a0760186574b0282f2f435e7",
    "https://example.com/docs/invoice.pdf": "9e107d9d372bb6826bd81d3542a419d6",
}


# A mock function to simulate fetching the file (no actual downloading is done)
def retrieve_document_content(url):
    # In a real case this would fetch content from `url`, but here we simulate
    if url in DOCUMENT_SIGNATURES:
        return "Document Content"
    else:
        raise IOError(f"Document at {url} not available")


def validate_document(file_url: str, crypt_signature: str) -> bool:
    """
    Confirm a document's integrity and authenticity by verifying its signature.

    This function:
      - Ensures the document is accessible (raises IOError if not).
      - Compares the provided signature against the expected signature for the URL.

    Args:
        file_url: The URL of the document to validate.
        crypt_signature: The provided cryptographic signature (hex string).

    Returns:
        True if the signature matches the expected signature for the document; False otherwise.

    Raises:
        IOError: If the document is inaccessible.
    """
    # Ensure document is accessible; propagate IOError if not accessible.
    retrieve_document_content(file_url)

    if not isinstance(file_url, str) or not isinstance(crypt_signature, str):
        return False

    expected = DOCUMENT_SIGNATURES.get(file_url)
    if expected is None:
        # If there's no known signature, treat as inaccessible (shouldn't happen due to retrieve).
        raise IOError(f"No known signature for document at {file_url}")

    # Normalize comparison to be case-insensitive on hex representation.
    return crypt_signature.strip().lower() == expected.strip().lower()


__all__ = [
    "fetch_document_content",
    "DOCUMENT_REGISTRY",
    "validate_signature",
    "DOCUMENT_SIGNATURES",
    "retrieve_document_content",
    "validate_document",
]
