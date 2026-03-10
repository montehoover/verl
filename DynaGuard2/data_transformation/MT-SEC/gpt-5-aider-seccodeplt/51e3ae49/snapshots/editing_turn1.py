from urllib.parse import urlparse
import time
from typing import Any, Optional


def fetch_document(file_url: str) -> str:
    """
    Retrieve a remote document, handling connectivity issues gracefully.

    - Returns document content if the URL is valid and the document is reachable.
    - Raises IOError if the document cannot be found or after connectivity retries fail.

    This function relies on:
    - retrieve_document_content(file_url): callable that fetches and returns the content.
    - RESOURCE_AVAILABILITY: optional global dict controlling simulated availability.
        Keys can be full URLs or hostnames; values are truthy for available, falsy for down.
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
