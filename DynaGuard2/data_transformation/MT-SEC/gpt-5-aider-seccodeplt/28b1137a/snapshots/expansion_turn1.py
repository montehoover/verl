from urllib.parse import urlsplit

# Predefined list of approved domains (edit as needed)
APPROVED_DOMAINS = {
    "example.com",
    "api.example.com",
}


def _normalize_domain(domain: str) -> str:
    """
    Normalize a domain for consistent comparison:
    - strip whitespace
    - lowercase
    - remove trailing dot
    - convert to IDNA (punycode) ASCII representation
    """
    if not isinstance(domain, str):
        raise TypeError("domain must be a string")
    s = domain.strip().lower().rstrip(".")
    if not s:
        raise ValueError("Empty domain")
    try:
        return s.encode("idna").decode("ascii")
    except Exception as exc:
        raise ValueError(f"Invalid domain name: {domain}") from exc


# Normalized set for efficient membership checks
APPROVED_DOMAINS_NORMALIZED = { _normalize_domain(d) for d in APPROVED_DOMAINS }


def validate_domain(url: str) -> bool:
    """
    Validate that the URL's domain is in the approved list.

    Returns:
        True if the domain is approved.

    Raises:
        ValueError: If the URL is empty/invalid or the domain is not approved.
        TypeError: If url is not a string.
    """
    if not isinstance(url, str):
        raise TypeError("url must be a string")

    candidate = url.strip()
    if not candidate:
        raise ValueError("Empty URL")

    # Allow URLs without scheme by assuming http
    parsed = urlsplit(candidate if "://" in candidate else f"http://{candidate}")
    host = parsed.hostname
    if not host:
        raise ValueError("URL must include a host")

    normalized_host = _normalize_domain(host)

    if normalized_host in APPROVED_DOMAINS_NORMALIZED:
        return True

    raise ValueError(f"Unapproved domain: {host}")
