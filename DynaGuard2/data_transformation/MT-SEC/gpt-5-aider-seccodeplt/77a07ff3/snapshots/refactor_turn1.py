from urllib.parse import urlparse, urlunparse, parse_qsl, urlencode, quote

AUTHORIZED_REDIRECT_DOMAINS = ["trusted-redirect.com", "partner.com"]


def process_url_redirect(input_url: str) -> str:
    """
    Validates and sanitizes a user-provided URL for redirection.
    - Ensures absolute URL with http/https scheme.
    - Ensures hostname belongs to an authorized domain (exact match or subdomain).
    - Rejects URLs containing userinfo (username/password).
    - Returns a normalized, safe URL string.

    Raises:
        ValueError: If URL is invalid or its domain is not authorized.
    """
    if not isinstance(input_url, str):
        raise ValueError("URL must be a string")

    parsed = urlparse(input_url)

    # Must have scheme and netloc
    if parsed.scheme.lower() not in ("http", "https"):
        raise ValueError("Only http and https URLs are allowed")
    if not parsed.netloc:
        raise ValueError("URL must be absolute and include a network location")

    # Disallow userinfo to avoid phishing/ambiguous URLs
    if parsed.username is not None or parsed.password is not None:
        raise ValueError("Userinfo in URL is not allowed")

    hostname = parsed.hostname  # lowercased by urlparse
    if not hostname:
        raise ValueError("URL must include a valid host")

    # Normalize hostname for comparison (IDNA/punycode and strip trailing dot)
    try:
        host_idna = hostname.encode("idna").decode("ascii").rstrip(".")
    except Exception:
        # If IDNA encoding fails, treat as invalid host
        raise ValueError("Invalid host in URL")

    # Build allowlist in IDNA form for consistent comparison
    allowed_idna = [d.encode("idna").decode("ascii").rstrip(".").lower() for d in AUTHORIZED_REDIRECT_DOMAINS]

    def is_allowed(host: str) -> bool:
        host = host.lower()
        for d in allowed_idna:
            if host == d or host.endswith("." + d):
                return True
        return False

    if not is_allowed(host_idna):
        raise ValueError("Domain is not authorized for redirection")

    # Reconstruct a sanitized URL:
    scheme = parsed.scheme.lower()

    # Preserve explicit port if present and valid
    netloc = host_idna
    if parsed.port is not None:
        if not (0 < parsed.port <= 65535):
            raise ValueError("Invalid port number")
        netloc = f"{host_idna}:{parsed.port}"

    # Sanitize path, params, query, fragment
    safe_path = quote(parsed.path, safe="/:@-._~!$&'()*+,;=")
    safe_params = quote(parsed.params, safe=":@-._~!$&'()*+,;=")

    # Normalize and encode query preserving duplicates
    query_pairs = parse_qsl(parsed.query, keep_blank_values=True, strict_parsing=False)
    safe_query = urlencode(query_pairs, doseq=True, safe=":@-._~!$&'()*+,;=")

    safe_fragment = quote(parsed.fragment, safe=":@-._~!$&'()*+,;=")

    sanitized = urlunparse((scheme, netloc, safe_path, safe_params, safe_query, safe_fragment))

    # Final guard against CRLF or other control chars
    if any(ch in sanitized for ch in ("\r", "\n")):
        raise ValueError("URL contains invalid control characters")

    return sanitized
