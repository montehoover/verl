from urllib.parse import urlparse, urlunparse

ALLOWED_REDIRECT_DOMAINS = ["trusted-redirect.com", "partner.com"]


def process_user_redirect(input_url: str) -> str:
    """
    Accepts a user-provided URL and returns a sanitized URL safe for redirection.
    Raises ValueError if the domain is not in ALLOWED_REDIRECT_DOMAINS or if the URL is invalid.
    """
    if not isinstance(input_url, str) or not input_url.strip():
        raise ValueError("Invalid URL: empty input")

    raw = input_url.strip()

    # Parse the URL
    parsed = urlparse(raw)

    # Handle scheme-relative URLs (e.g., //example.com/path) by defaulting to https
    if not parsed.scheme and parsed.netloc:
        scheme = "https"
    else:
        scheme = (parsed.scheme or "").lower()

    # Only allow HTTP(S) redirects
    if scheme not in ("http", "https"):
        raise ValueError("Invalid URL: unsupported scheme")

    # Ensure a hostname exists
    hostname = parsed.hostname  # excludes credentials and brackets
    if not hostname:
        raise ValueError("Invalid URL: missing hostname")

    host_l = hostname.lower().rstrip(".")

    # Check against allowed domains (exact match or subdomain)
    def is_allowed(host: str) -> bool:
        for d in ALLOWED_REDIRECT_DOMAINS:
            dl = d.lower().rstrip(".")
            if host == dl or host.endswith("." + dl):
                return True
        return False

    if not is_allowed(host_l):
        raise ValueError("Invalid URL: domain not allowed")

    # Validate port if present
    port = parsed.port
    if port is not None:
        if port <= 0 or port > 65535:
            raise ValueError("Invalid URL: invalid port")

    # IDNA-encode the host to ASCII
    try:
        safe_host = host_l.encode("idna").decode("ascii")
    except Exception:
        raise ValueError("Invalid URL: invalid hostname")

    # Build sanitized netloc (exclude any userinfo)
    if port is not None:
        default_port = (scheme == "http" and port == 80) or (scheme == "https" and port == 443)
        netloc = safe_host if default_port else f"{safe_host}:{port}"
    else:
        netloc = safe_host

    # Remove control characters from path/query/params
    def strip_ctrl(s: str) -> str:
        return "".join(ch for ch in s if ch >= " " and ch != "\x7f")

    path = strip_ctrl(parsed.path or "/")
    if not path.startswith("/"):
        path = "/" + path

    params = strip_ctrl(parsed.params or "")
    query = strip_ctrl(parsed.query or "")

    # Drop fragment entirely
    fragment = ""

    sanitized = urlunparse((scheme, netloc, path, params, query, fragment))
    return sanitized
