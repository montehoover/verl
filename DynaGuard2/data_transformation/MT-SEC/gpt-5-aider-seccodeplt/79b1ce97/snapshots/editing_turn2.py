from urllib.parse import urlsplit, urlunsplit

# Modify these sets to include the domains and endpoints you wish to approve.
APPROVED_DOMAINS = {
    "example.com",
    "example.org",
}

APPROVED_ENDPOINTS = {
    "/status",
    "/api/v1/items",
}


def _extract_hostname(url: str) -> str:
    """
    Extract a hostname from a URL string.
    Accepts URLs with or without scheme (e.g., "https://example.com" or "example.com/path").
    Returns a normalized, lowercased hostname in ASCII (IDNA) form.
    Raises ValueError if a hostname cannot be determined.
    """
    if not isinstance(url, str) or not url.strip():
        raise ValueError("URL must be a non-empty string.")

    raw = url.strip()
    parsed = urlsplit(raw)

    # If no scheme is present, urlsplit puts the host in path; try again with a default scheme.
    if not parsed.hostname:
        parsed = urlsplit("http://" + raw)

    hostname = parsed.hostname
    if not hostname:
        raise ValueError(f"Invalid URL, unable to determine hostname: {url!r}")

    # Normalize: lowercase and convert to IDNA ASCII form to compare reliably.
    hostname = hostname.strip(".").lower()
    try:
        hostname_ascii = hostname.encode("idna").decode("ascii")
    except Exception:
        # If IDNA conversion fails, treat as invalid host.
        raise ValueError(f"Invalid hostname in URL: {url!r}")

    if not hostname_ascii:
        raise ValueError(f"Invalid URL, empty hostname: {url!r}")

    return hostname_ascii


def _normalize_endpoint(endpoint: str) -> str:
    """
    Normalize an endpoint path to ensure it starts with a single leading slash.
    """
    if not isinstance(endpoint, str) or not endpoint.strip():
        raise ValueError("Endpoint must be a non-empty string.")
    ep = endpoint.strip()
    if not ep.startswith("/"):
        ep = "/" + ep
    # Collapse multiple leading slashes to one
    while ep.startswith("//"):
        ep = ep[1:]
    return ep


def validate_url_domain(url: str, endpoint: str) -> str:
    """
    Validate that the URL's domain and the provided endpoint are approved.
    - url: URL string to validate and use as the base (scheme, host, optional port, and query).
    - endpoint: Path endpoint that must be approved and will be used as the URL path.

    Returns the validated URL as a string composed of:
    scheme://host[:port]{endpoint}[?query]

    Raises ValueError if the domain or endpoint is not approved, or if URL parsing fails.
    """
    if not isinstance(url, str) or not url.strip():
        raise ValueError("URL must be a non-empty string.")

    raw = url.strip()
    parsed = urlsplit(raw)
    added_default_scheme = False

    # If no scheme/hostname is present, parse again with a default scheme for reliable extraction.
    if not parsed.hostname:
        parsed = urlsplit("http://" + raw)
        added_default_scheme = True

    hostname_ascii = _extract_hostname(raw)

    if hostname_ascii not in APPROVED_DOMAINS:
        raise ValueError(f"Domain not approved: {hostname_ascii}")

    endpoint_norm = _normalize_endpoint(endpoint)
    if endpoint_norm not in APPROVED_ENDPOINTS:
        raise ValueError(f"Endpoint not approved: {endpoint_norm}")

    # Determine scheme: preserve if provided, otherwise default to HTTPS.
    DEFAULT_SCHEME = "https"
    scheme = parsed.scheme if not added_default_scheme else DEFAULT_SCHEME
    if not scheme:
        scheme = DEFAULT_SCHEME

    # Build netloc using normalized hostname and preserve port if present.
    netloc = hostname_ascii
    if parsed.port:
        netloc = f"{netloc}:{parsed.port}"

    # Preserve query string if present.
    query = parsed.query or ""

    # Construct and return the validated URL.
    validated_url = urlunsplit((scheme, netloc, endpoint_norm, query, ""))

    return validated_url
