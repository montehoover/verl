"""
SSRF-safe API endpoint validation utilities.

This module provides helper functions and a single entry-point `enforce_api_security`
to validate that a given API endpoint URL:
  - Uses HTTPS only.
  - Does not include inline credentials (username/password).
  - Does not target internal addresses (localhost, loopback, common RFC1918 ranges).
  - Resolves to an explicitly allowed external domain (or its subdomains).

The logic for checking internal hosts and allowed domains is separated into pure
utility functions to improve testability and maintainability.
"""

import re
from urllib.parse import urlparse

# Explicit list of approved external API domains.
ALLOWED_API_DOMAINS = ["api.trustedsource.com", "gateway.partnerapi.com"]

# Regex patterns describing internal/unsafe hosts we block to mitigate SSRF.
# Note: This list is not exhaustive of all private or link-local ranges; extend as needed.
INTERNAL_IP_PATTERNS = [
    r"^localhost$",                      # Localhost
    r"^127\.0\.0\.\d+$",                 # Loopback address
    r"^10\.\d{1,3}\.\d{1,3}\.\d{1,3}$",  # Private 10.x.x.x range
    r"^192\.168\.\d{1,3}\.\d{1,3}$"      # Private 192.168.x.x range
]

# Pre-compile regex patterns once for efficiency.
COMPILED_INTERNAL_IP_PATTERNS = [re.compile(p) for p in INTERNAL_IP_PATTERNS]


def is_internal_host(host: str) -> bool:
    """
    Determine whether the given host matches internal network patterns.

    This function considers a host "internal" if it matches any of the patterns
    defined in INTERNAL_IP_PATTERNS (e.g., localhost, loopback, private ranges).

    Args:
        host: A hostname or IPv4 address string. The value is normalized by
              lowercasing and trimming trailing dots.

    Returns:
        True if the host is considered internal; False otherwise.

    Examples:
        is_internal_host("localhost") -> True
        is_internal_host("127.0.0.1") -> True
        is_internal_host("10.1.2.3")  -> True
        is_internal_host("192.168.1.10") -> True
        is_internal_host("api.trustedsource.com") -> False
    """
    normalized_host = (host or "").lower().rstrip(".")
    for pattern in COMPILED_INTERNAL_IP_PATTERNS:
        if pattern.match(normalized_host):
            return True
    return False


def is_allowed_domain(host: str, allowed_domains=None) -> bool:
    """
    Check whether the given host is within the allowed API domains (exact or subdomain).

    A host is considered allowed if it matches exactly one of the entries in
    `allowed_domains` or is a subdomain of one of those entries. For example,
    "sub.api.trustedsource.com" is allowed if "api.trustedsource.com" is in the list.

    Args:
        host: A hostname to check.
        allowed_domains: Optional list of allowed base domains. Defaults to
                         ALLOWED_API_DOMAINS when omitted or None.

    Returns:
        True if the host is allowed; False otherwise.

    Examples:
        is_allowed_domain("api.trustedsource.com") -> True
        is_allowed_domain("eu.api.trustedsource.com") -> True
        is_allowed_domain("malicious.com") -> False
    """
    if allowed_domains is None:
        allowed_domains = ALLOWED_API_DOMAINS

    normalized_host = (host or "").lower().rstrip(".")
    if not normalized_host:
        return False

    for domain in allowed_domains:
        d = (domain or "").lower().rstrip(".")
        if not d:
            continue
        if normalized_host == d or normalized_host.endswith("." + d):
            return True
    return False


def _internal_host_reason(host: str) -> str:
    """
    Provide a human-readable explanation for why a host is considered internal.

    Args:
        host: The normalized host string.

    Returns:
        A short description such as 'localhost', 'loopback (127.0.0.0/8)',
        'private (10.0.0.0/8)', 'private (192.168.0.0/16)', or 'internal network'.
    """
    h = (host or "").lower().rstrip(".")
    if h == "localhost":
        return "localhost"
    if h.startswith("127."):
        return "loopback (127.0.0.0/8)"
    if h.startswith("10."):
        return "private (10.0.0.0/8)"
    if h.startswith("192.168."):
        return "private (192.168.0.0/16)"
    return "internal network"


def enforce_api_security(api_endpoint: str) -> str:
    """
    Validate that the given URL points to a secure, external API endpoint.

    This function enforces the following rules in order:
      1) The URL must be a non-empty string.
      2) The URL must use the 'https' scheme.
      3) The URL must contain a valid host and must not include credentials.
      4) The host must not be an internal address (e.g., localhost, loopback, private ranges).
      5) The host must belong to the allowed API domains list (exact or subdomain match).

    Args:
        api_endpoint: The URL string to validate.

    Returns:
        The original (stripped) URL if it passes all checks.

    Raises:
        ValueError: If the input is not a string, is empty/invalid, uses a non-HTTPS scheme,
                    includes credentials, targets an internal address, or is not in the
                    approved domain list.

    Examples:
        enforce_api_security("https://api.trustedsource.com/v1/resource")
        enforce_api_security("https://eu.api.trustedsource.com/v2/items?id=123")
    """
    # 1) Basic type and emptiness checks
    if not isinstance(api_endpoint, str):
        raise ValueError(f"URL must be a string; received type '{type(api_endpoint).__name__}'.")
    url = api_endpoint.strip()
    if not url:
        raise ValueError("URL cannot be empty or whitespace.")

    parsed = urlparse(url)

    # 2) Enforce HTTPS
    scheme = (parsed.scheme or "").lower()
    if scheme != "https":
        raise ValueError(f"URL must use HTTPS; received scheme '{parsed.scheme or ''}'.")

    # 3) Validate netloc/host and disallow credentials
    if not parsed.netloc:
        raise ValueError("URL must include a network location (host).")
    if parsed.username is not None or parsed.password is not None:
        raise ValueError("Credentials in URL are not allowed (username/password detected).")

    host = (parsed.hostname or "").lower().rstrip(".")
    if not host:
        raise ValueError("URL must contain a valid hostname.")

    # 4) Block internal hosts
    if is_internal_host(host):
        reason = _internal_host_reason(host)
        raise ValueError(f"Requests to internal addresses are not allowed: host '{host}' is {reason}.")

    # 5) Restrict to allowed external domains
    if not is_allowed_domain(host):
        allowed_list = ", ".join(ALLOWED_API_DOMAINS)
        raise ValueError(f"Unapproved domain: host '{host}' is not in the allowed list [{allowed_list}].")

    return url
