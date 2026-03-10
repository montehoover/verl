from typing import Dict, Optional

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


__all__ = ["fetch_document_content", "DOCUMENT_REGISTRY"]
