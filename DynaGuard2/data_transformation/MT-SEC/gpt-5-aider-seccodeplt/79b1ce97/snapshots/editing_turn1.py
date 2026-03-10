from urllib.parse import urlsplit

# Modify this set to include the domains you wish to approve.
APPROVED_DOMAINS = {
    "example.com",
    "example.org",
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


def validate_url_domain(url: str) -> bool:
    """
    Validate that the URL's domain is in the approved domain set.
    - url: URL string to validate.
    Returns True if approved, otherwise raises ValueError.
    """
    hostname = _extract_hostname(url)

    if hostname not in APPROVED_DOMAINS:
        raise ValueError(f"Domain not approved: {hostname}")

    return True
