from urllib.parse import urlparse
import requests

def process_webhook(webhook_url: str) -> str:
    """
    Validate and normalize a webhook URL.

    - Ensures the URL uses HTTPS (adds https if scheme is missing).
    - Performs a HEAD request to verify reachability (falls back to GET on 405).
    - Returns the normalized URL string on success.
    - Raises ValueError if the URL is not secure or unreachable.

    Args:
        webhook_url: The webhook URL as a string.

    Returns:
        The normalized HTTPS URL.
    """
    if not isinstance(webhook_url, str) or not webhook_url.strip():
        raise ValueError("Webhook URL must be a non-empty string.")

    original = webhook_url.strip()
    parsed = urlparse(original)

    # Support schemeless inputs like "example.com/path"
    if not parsed.scheme and not parsed.netloc and parsed.path:
        parsed_fallback = urlparse("//" + original)
        if parsed_fallback.netloc:
            parsed = parsed_fallback

    scheme = (parsed.scheme or "").lower()

    # If no scheme is provided, default to https for normalization.
    if not scheme:
        scheme = "https"

    if scheme != "https":
        raise ValueError("Webhook URL must use HTTPS.")

    host = (parsed.hostname or "").lower()
    if not host:
        raise ValueError("Webhook URL must include a valid host.")

    # Determine port and build netloc, omitting default 443
    port = parsed.port
    is_ipv6 = ":" in host and not host.startswith("[")
    bracketed_host = f"[{host}]" if is_ipv6 else host
    if port and port != 443:
        netloc = f"{bracketed_host}:{port}"
    else:
        netloc = bracketed_host

    # Normalize path and query; remove fragment
    path = parsed.path or "/"
    if not path.startswith("/"):
        path = "/" + path
    query = parsed.query

    normalized_url = f"{scheme}://{netloc}{path}"
    if query:
        normalized_url += f"?{query}"

    # Verify reachability via HEAD, follow redirects
    try:
        resp = requests.head(normalized_url, allow_redirects=True, timeout=5)
        # Some endpoints may not allow HEAD; try GET as a fallback on 405
        if resp.status_code == 405:
            get_resp = requests.get(normalized_url, allow_redirects=True, timeout=5, stream=True)
            # We consider any HTTP response as "reachable" (network available)
            get_resp.close()
    except requests.RequestException as exc:
        raise ValueError(f"Webhook URL is unreachable: {exc}") from exc

    return normalized_url
