import re
import ipaddress
from urllib.parse import urlsplit
from typing import Dict


_SCHEME_RE = re.compile(r"^[A-Za-z][A-Za-z0-9+.-]*$")


def _is_valid_scheme(scheme: str) -> bool:
    if not scheme or not isinstance(scheme, str):
        return False
    return _SCHEME_RE.match(scheme) is not None


def _is_valid_domain(host: str) -> bool:
    if not host or not isinstance(host, str):
        return False

    # Try IP literal (both IPv4 and IPv6)
    try:
        ipaddress.ip_address(host)
        return True
    except ValueError:
        pass

    # Validate IDN/hostname by converting to IDNA (punycode) then checking labels
    try:
        host_idna = host.encode("idna").decode("ascii")
    except UnicodeError:
        return False

    if len(host_idna) > 253:
        return False

    labels = host_idna.split(".")
    for label in labels:
        if not label or len(label) > 63:
            return False
        # Must start/end alphanumeric; can contain hyphens inside
        if not re.fullmatch(r"[A-Za-z0-9](?:[A-Za-z0-9-]*[A-Za-z0-9])?", label):
            return False

    return True


def parse_and_validate_url(url: str) -> Dict[str, str]:
    """
    Parse and validate a URL.

    Args:
        url: The URL string to parse.

    Returns:
        A dict with at least the keys: 'scheme', 'domain', and 'path'.

    Raises:
        ValueError: If the URL is invalid.
    """
    if not isinstance(url, str):
        raise ValueError("URL must be a string")

    url = url.strip()
    if not url:
        raise ValueError("URL must not be empty")

    parsed = urlsplit(url)

    # Validate scheme
    if not _is_valid_scheme(parsed.scheme):
        raise ValueError("Invalid or missing URL scheme")

    # Extract hostname (domain) and validate
    host = parsed.hostname
    if host is None or not _is_valid_domain(host):
        raise ValueError("Invalid or missing URL domain/host")

    # Validate port if present (urlsplit.port may raise ValueError for bad ports)
    try:
        _ = parsed.port  # Accessing triggers validation
    except ValueError:
        raise ValueError("Invalid port in URL")

    # Build result dictionary
    result = {
        "scheme": parsed.scheme.lower(),
        "domain": host,
        "path": parsed.path or "",
    }

    return result
