from urllib.parse import urlparse


# Hardcoded list of trusted domains (normalize to lowercase, no trailing dots)
TRUSTED_DOMAINS = {
    "example.com",
    "mycompany.com",
    "trusted.org",
}


def _normalize_hostname(host: str) -> str:
    # Lowercase and remove any trailing dot for consistent comparisons
    return host.strip().lower().rstrip(".")


def extract_domain(url: str) -> str:
    """
    Extract the domain (host) from a URL string.

    Examples:
      - "https://www.example.com/path" -> "www.example.com"
      - "example.com/path" -> "example.com"
      - "http://user:pass@example.com:8080" -> "example.com"
      - "http://[2001:db8::1]:8080/" -> "2001:db8::1"

    Returns an empty string if no domain can be determined.
    """
    if not isinstance(url, str) or not url:
        return ""

    parsed = urlparse(url)

    # Handle URLs without a scheme like "example.com/path"
    if not parsed.netloc and parsed.path:
        parsed = urlparse("//" + url)

    host = parsed.hostname  # Strips credentials and port; lowercases host
    if not host:
        return ""

    # Remove any trailing dot from the hostname
    if host.endswith("."):
        host = host[:-1]

    return host


# Pre-normalize trusted domains for faster comparisons
_NORMALIZED_TRUSTED = {_normalize_hostname(d) for d in TRUSTED_DOMAINS if d}


def is_trusted_domain(url: str) -> bool:
    """
    Determine whether the URL's domain is trusted.

    A domain is considered trusted if it exactly matches a trusted domain
    or is a subdomain of a trusted domain.

    Examples:
      - "https://sub.example.com/page" with "example.com" trusted -> True
      - "https://example.com" with "example.com" trusted -> True
      - "https://nottrusted.io" -> False
    """
    domain = extract_domain(url)
    if not domain:
        return False

    domain = _normalize_hostname(domain)
    for trusted in _NORMALIZED_TRUSTED:
        if domain == trusted or domain.endswith("." + trusted):
            return True
    return False
