import urllib.request
import urllib.error


_USER_AGENT = "document-exists-check/1.0 (+https://example.com)"


def document_exists(doc_link: str) -> bool:
    """
    Check if a document is available at the given URL.

    Args:
        doc_link: The URL to check.

    Returns:
        True if the document is found (HTTP 2xx), False if not found (HTTP 404/410).

    Raises:
        IOError: If the document cannot be accessed due to network issues, permission errors,
                 server errors, invalid URL, or other non-2xx/404/410 HTTP responses.
    """
    timeout = 10.0
    headers = {
        "User-Agent": _USER_AGENT,
        "Accept": "*/*",
    }

    # Try a HEAD request first to avoid downloading the body.
    head_req = urllib.request.Request(doc_link, method="HEAD", headers=headers)

    try:
        with urllib.request.urlopen(head_req, timeout=timeout) as resp:
            code = resp.getcode()
            # Any 2xx means the resource exists
            if 200 <= code < 300:
                return True
            # In practice, urlopen follows redirects automatically; non-2xx shouldn't reach here.
            raise IOError(f"Unexpected HTTP status {code} on HEAD for {doc_link}")
    except urllib.error.HTTPError as e:
        # Not found
        if e.code in (404, 410):
            return False
        # HEAD not allowed -> fallback to GET
        if e.code == 405:
            pass  # handled by GET fallback below
        else:
            # Access/other errors (auth required, forbidden, rate-limited, server error, etc.)
            raise IOError(f"Cannot access document at {doc_link}: HTTP {e.code}") from e
    except urllib.error.URLError as e:
        raise IOError(f"Cannot access document at {doc_link}: {e.reason}") from e
    except ValueError as e:
        # Malformed URL
        raise IOError(f"Invalid URL {doc_link}: {e}") from e
    else:
        # HEAD succeeded and already returned; this else won't run,
        # but keep structure explicit for clarity.
        return True

    # Fallback: Use a minimal GET to probe existence (for servers that disallow HEAD).
    get_headers = dict(headers)
    # Ask for only the first byte to minimize transfer; servers may respond 206 or 200.
    get_headers["Range"] = "bytes=0-0"
    get_req = urllib.request.Request(doc_link, method="GET", headers=get_headers)

    try:
        with urllib.request.urlopen(get_req, timeout=timeout) as resp:
            code = resp.getcode()
            # 2xx includes 200/204; 206 Partial Content also indicates existence.
            if (200 <= code < 300) or code == 206:
                return True
            if code in (404, 410):
                return False
            # Some servers may return 416 for zero-length resources with a Range request.
            if code == 416:
                return True
            raise IOError(f"Cannot access document at {doc_link}: HTTP {code}")
    except urllib.error.HTTPError as e:
        if e.code in (404, 410):
            return False
        if e.code == 416:
            # Requested Range Not Satisfiable can indicate the resource exists but is empty.
            return True
        raise IOError(f"Cannot access document at {doc_link}: HTTP {e.code}") from e
    except urllib.error.URLError as e:
        raise IOError(f"Cannot access document at {doc_link}: {e.reason}") from e
    except ValueError as e:
        raise IOError(f"Invalid URL {doc_link}: {e}") from e
