from urllib.parse import urlsplit
import ipaddress
from typing import Set

# Predefined list of allowed domains/hosts.
# Update this set as needed for your application.
ALLOWED_DOMAINS = {
    "example.com",
    "api.example.com",
    "localhost",
    "127.0.0.1",
}

def _normalize_host_for_compare(host: str) -> str:
    """
    Normalize a hostname or IP for comparison:
    - For IP addresses (v4/v6), return the compressed canonical form.
    - For domain names, return lower-case ASCII using IDNA (punycode) and strip any trailing dot.
    """
    if not isinstance(host, str):
        return ""

    host = host.strip()
    if not host:
        return ""

    # Try IP normalization
    try:
        ip = ipaddress.ip_address(host)
        return ip.compressed
    except ValueError:
        pass

    # Domain normalization: remove trailing dot (FQDN) and convert to ASCII (IDNA), lower-case
    domain = host.rstrip(".")
    try:
        domain_ascii = domain.encode("idna").decode("ascii")
    except UnicodeError:
        # If IDNA encoding fails, fallback to lower-case as-is
        domain_ascii = domain
    return domain_ascii.lower()

# Pre-normalize allowed hosts for efficient comparison
_ALLOWED_HOSTS: Set[str] = {_normalize_host_for_compare(h) for h in ALLOWED_DOMAINS}

def validate_url_domain(url: str) -> bool:
    """
    Validate that the URL is well-formed and that its hostname is in the allowed list.

    Args:
        url: The URL string to validate.

    Returns:
        True if the URL's hostname is in the allowed list, otherwise False.

    Raises:
        ValueError: If the URL is invalid (e.g., missing hostname).
    """
    if not isinstance(url, str) or not url.strip():
        raise ValueError("Invalid URL: must be a non-empty string")

    parts = urlsplit(url)

    # urlsplit ensures hostname is parsed only for URLs with a network location
    host = parts.hostname
    if not host:
        raise ValueError("Invalid URL: missing hostname")

    normalized_host = _normalize_host_for_compare(host)
    if not normalized_host:
        raise ValueError("Invalid URL: could not normalize hostname")

    return normalized_host in _ALLOWED_HOSTS
