from urllib.parse import urlsplit
import ipaddress
import re
from typing import Set

# Predefined set of allowed domains (edit this set to fit your allowlist)
ALLOWED_DOMAINS: Set[str] = {
    "example.com",
    "example.org",
}

_LABEL_RE = re.compile(r"^[a-z0-9-]+$", re.IGNORECASE)


def _canonicalize_hostname(host: str) -> str:
    """
    Convert a hostname to a canonical ASCII (IDNA) lowercase form without a trailing dot.
    Returns an empty string if host is falsy.
    """
    if not host:
        return ""
    h = host.strip().rstrip(".").lower()
    # Convert Unicode domains to ASCII punycode for consistent comparison
    try:
        h_ascii = h.encode("idna").decode("ascii")
    except Exception:
        # If IDNA encoding fails, keep the original; downstream validation may reject it
        h_ascii = h
    return h_ascii


def _is_valid_hostname_ascii(host: str) -> bool:
    """
    Basic validation for an ASCII hostname (not an IP).
    - Total length <= 253
    - Each label 1..63 chars, alnum or hyphen, not starting/ending with hyphen
    """
    if not host:
        return False
    if len(host) > 253:
        return False
    labels = host.split(".")
    for label in labels:
        if not label:
            return False
        if len(label) > 63:
            return False
        if label[0] == "-" or label[-1] == "-":
            return False
        if not _LABEL_RE.fullmatch(label):
            return False
    return True


# Canonicalize allowed domains up-front
_ALLOWED_DOMAINS_CANON: Set[str] = {_canonicalize_hostname(d) for d in ALLOWED_DOMAINS if d}


def validate_url_domain(url: str) -> bool:
    """
    Validate a URL and check if its domain is in the allowed list.

    - Returns True if the URL is valid and its hostname matches an allowed domain
      (either exact match or a subdomain of an allowed domain).
    - Returns False if the URL is valid but its hostname does not match the allowlist.
    - Raises ValueError if the URL itself is invalid.

    Only http and https schemes are considered valid for this check.
    """
    if not isinstance(url, str):
        raise ValueError("Invalid URL")

    candidate = url.strip()
    if not candidate:
        raise ValueError("Invalid URL")

    try:
        parts = urlsplit(candidate)
    except Exception as e:
        raise ValueError("Invalid URL") from e

    if parts.scheme not in ("http", "https"):
        raise ValueError("Invalid URL")

    if not parts.netloc:
        raise ValueError("Invalid URL")

    # Validate port if present (accessing .port triggers validation in urllib.parse)
    try:
        _ = parts.port  # noqa: F841
    except Exception as e:
        raise ValueError("Invalid URL") from e

    host = parts.hostname
    if not host:
        raise ValueError("Invalid URL")

    # If host is an IP address, URL is valid but IPs are not domain-allowed
    try:
        ipaddress.ip_address(host)
        return False
    except ValueError:
        pass  # Not an IP, proceed with domain validation

    host_canon = _canonicalize_hostname(host)
    if not _is_valid_hostname_ascii(host_canon):
        raise ValueError("Invalid URL")

    # Match exact allowed domain or any of its subdomains
    for allowed in _ALLOWED_DOMAINS_CANON:
        if host_canon == allowed or host_canon.endswith("." + allowed):
            return True

    return False
