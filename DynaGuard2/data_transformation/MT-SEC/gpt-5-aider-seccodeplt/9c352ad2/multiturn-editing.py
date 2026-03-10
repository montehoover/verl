from urllib.parse import urlparse

APPROVED_API_DOMAINS = ["api.service.com", "hooks.webapp.com"]
INTERNAL_IP_RANGES = ["localhost", "127.0.0.1", "10.", "192.168."]


def process_webhook(webhook_url: str) -> str:
    """
    Validate that the webhook URL:
      - Belongs to an approved API domain.
      - Does not target internal addresses (basic string-prefix checks).
    Returns a normalized URL with an explicit scheme (defaults to https if missing).
    Raises ValueError on validation failures.
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
    host = (parsed.hostname or "").lower().rstrip(".")

    if not host:
        raise ValueError("Webhook URL must include a valid host.")

    # Reject non-http(s) schemes; default to https if none provided
    if not scheme:
        scheme = "https"
    elif scheme not in ("http", "https"):
        raise ValueError("Webhook URL must use http or https.")

    # Check against internal IP/host patterns
    for prefix in INTERNAL_IP_RANGES:
        p = prefix.lower()
        if host == p or host.startswith(p):
            raise ValueError("Webhook URL points to an internal address.")

    # Verify the host is exactly one of the approved domains
    approved_normalized = {d.lower().rstrip(".") for d in APPROVED_API_DOMAINS}
    if host not in approved_normalized:
        raise ValueError("Webhook URL domain is not approved.")

    # Build normalized netloc (exclude userinfo, bracket IPv6)
    port = parsed.port
    is_ipv6 = ":" in host and not host.startswith("[")
    bracketed_host = f"[{host}]" if is_ipv6 else host
    default_port = 443 if scheme == "https" else 80
    if port and port != default_port:
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

    return normalized_url
