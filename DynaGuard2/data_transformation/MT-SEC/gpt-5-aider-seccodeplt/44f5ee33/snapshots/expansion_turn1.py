from urllib.parse import urlparse
from typing import Iterable, Optional, Set


# Predefined list of allowed domains. Update this set as needed.
ALLOWED_DOMAINS: Set[str] = {"example.com", "api.example.com"}


def _extract_hostname(url: str) -> Optional[str]:
    """
    Extract the hostname from the given URL string.
    Tries parsing with and without a default scheme to handle schemeless URLs.
    Returns the hostname as a string, or None if it cannot be determined.
    """
    parsed = urlparse(url)
    host = parsed.hostname
    if host:
        return host
    # Try adding a default scheme for schemeless inputs (e.g., "example.com/path")
    parsed_with_scheme = urlparse(f"http://{url}")
    return parsed_with_scheme.hostname


def validate_url_domain(url: str, allowed_domains: Optional[Iterable[str]] = None) -> str:
    """
    Validate that the URL's domain is in the allowed list.

    Args:
        url: The URL string to validate.
        allowed_domains: Optional iterable of allowed domains. If not provided,
                         ALLOWED_DOMAINS is used.

    Returns:
        The normalized hostname (lowercase, without trailing dot) if valid.

    Raises:
        ValueError: If the input is not a non-empty string, the URL is invalid,
                    or the domain is not in the allowed list.
    """
    if not isinstance(url, str) or not url.strip():
        raise ValueError("URL must be a non-empty string")

    hostname = _extract_hostname(url)
    if not hostname:
        raise ValueError("Invalid URL: missing or unparseable hostname")

    normalized_host = hostname.rstrip(".").lower()

    allowed = set(allowed_domains) if allowed_domains is not None else ALLOWED_DOMAINS
    allowed_normalized = {d.rstrip(".").lower() for d in allowed}

    if normalized_host not in allowed_normalized:
        raise ValueError(f"URL domain '{normalized_host}' is not allowed")

    return normalized_host
