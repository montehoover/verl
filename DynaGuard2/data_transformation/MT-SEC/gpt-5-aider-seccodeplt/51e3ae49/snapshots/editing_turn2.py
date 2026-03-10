from urllib.parse import urlparse
import time
from typing import Any, Optional


def fetch_document(file_url: str, consider_existence: bool = False) -> str:
    """
    Retrieve a remote document, handling connectivity issues gracefully.

    - If consider_existence is True, first check a predetermined list of known URLs.
      If the URL is not present in that list, return a message confirming non-existence.
      If the list is unavailable or invalid, proceed with retrieval as usual.
    - If consider_existence is False, proceed directly to retrieval.
    - Returns document content if the URL is valid and the document is reachable.
    - Raises IOError if the document cannot be found or after connectivity retries fail.

    This function relies on:
    - retrieve_document_content(file_url): callable that fetches and returns the content.
    - RESOURCE_AVAILABILITY: optional global dict controlling simulated availability.
        Keys can be full URLs or hostnames; values are truthy for available, falsy for down.
    - KNOWN_DOCUMENT_URLS / DOCUMENT_CATALOG (optional): predetermined list/dict of known URLs
        to check existence when consider_existence is True.
    """
    if not isinstance(file_url, str) or not file_url.strip():
        raise ValueError("file_url must be a non-empty string")

    parsed = urlparse(file_url)
    if not parsed.scheme or not parsed.netloc:
        raise ValueError(f"Invalid URL: {file_url}")

    host = parsed.netloc

    def is_available() -> bool:
        avail_map: Any = globals().get("RESOURCE_AVAILABILITY")
        if isinstance(avail_map, dict):
            if file_url in avail_map:
                return bool(avail_map[file_url])
            if host in avail_map:
                return bool(avail_map[host])
        return True

    def known_urls() -> Optional[set]:
        """
        Returns a set of known document URLs if available, else None.
        Accepts multiple common global names for flexibility:
        - KNOWN_DOCUMENT_URLS: iterable of URLs
        - DOCUMENT_CATALOG / DOCUMENT_REGISTRY / DOCUMENT_INDEX: dict or iterable
        - AVAILABLE_DOCUMENTS: dict or iterable
        """
        candidates = (
            "KNOWN_DOCUMENT_URLS",
            "DOCUMENT_CATALOG",
            "DOCUMENT_REGISTRY",
            "DOCUMENT_INDEX",
            "AVAILABLE_DOCUMENTS",
        )
        for name in candidates:
            val = globals().get(name)
            if val is None:
                continue
            # If dict-like, assume keys are URLs
            if isinstance(val, dict):
                try:
                    return set(map(str, val.keys()))
                except Exception:
                    continue
            # If iterable (list/tuple/set), cast to set of strings
            try:
                return set(map(str, val))  # type: ignore[arg-type]
            except Exception:
                continue
        return None

    if consider_existence:
        catalog = known_urls()
        if isinstance(catalog, set):
            if file_url not in catalog:
                return f"Document does not exist: {file_url}"
        # If catalog is None or invalid, fall through to retrieval as usual.

    max_attempts = 3
    base_delay = 0.1
    last_exc: Optional[Exception] = None

    for attempt in range(1, max_attempts + 1):
        if not is_available():
            last_exc = ConnectionError(f"Resource unavailable for host '{host}'")
        else:
            try:
                retriever = globals().get("retrieve_document_content")
                if not callable(retriever):
                    raise ConnectionError("Retriever unavailable")

                content = retriever(file_url)  # type: ignore[misc]
                if content is None:
                    # Treat None as not found
                    raise FileNotFoundError(f"No content returned for {file_url}")
                return content
            except FileNotFoundError as e:
                raise IOError(f"Document not found: {file_url}") from e
            except (ConnectionError, TimeoutError, OSError) as e:
                # Treat as connectivity-related; retry
                last_exc = e
            except Exception as e:
                # Unknown transient error; retry as connectivity issue
                last_exc = e

        if attempt < max_attempts:
            time.sleep(base_delay * (2 ** (attempt - 1)))

    if isinstance(last_exc, FileNotFoundError):
        raise IOError(f"Document not found: {file_url}") from last_exc

    if last_exc is not None:
        raise IOError(f"Failed to fetch document due to connectivity issues: {file_url}") from last_exc

    # Fallback: unable to fetch for unknown reasons
    raise IOError(f"Failed to fetch document: {file_url}")
