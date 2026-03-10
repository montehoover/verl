import re
import ipaddress
from urllib.parse import urlsplit

ALLOWED_SCHEMES = {"http", "https"}


def _is_valid_host(hostname: str) -> bool:
    if hostname is None or hostname == "":
        return False

    # Check if hostname is an IP address (IPv4 or IPv6)
    try:
        ipaddress.ip_address(hostname)
        return True
    except ValueError:
        pass

    # Validate domain name (supports internationalized domains via IDNA)
    # Allow trailing dot for FQDNs in input, but validate without it
    if hostname.endswith("."):
        hostname = hostname[:-1]

    try:
        ascii_host = hostname.encode("idna").decode("ascii")
    except UnicodeError:
        return False

    if len(ascii_host) > 253:
        return False

    label_re = re.compile(r"^[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?$")
    labels = ascii_host.split(".")
    for label in labels:
        if not label or len(label) > 63:
            return False
        if not label_re.match(label):
            return False

    return True


def validate_url(url: str) -> bool:
    """
    Validate that the given URL is an absolute HTTP(S) URL with a valid host and optional port.
    Returns True when valid; raises ValueError otherwise.
    """
    if not isinstance(url, str):
        raise ValueError("URL must be a string")

    url = url.strip()
    if not url:
        raise ValueError("URL must not be empty")

    # Disallow control characters anywhere in the URL
    if any(ord(ch) < 32 for ch in url):
        raise ValueError("URL contains control characters")

    parts = urlsplit(url)

    # Validate scheme
    scheme = (parts.scheme or "").lower()
    if scheme not in ALLOWED_SCHEMES:
        raise ValueError(f"Unsupported URL scheme: {parts.scheme!r}")

    # Must have a network location (host[:port])
    if not parts.netloc:
        raise ValueError("URL must include a network location (host)")

    # Disallow userinfo (credentials) in URLs for security
    if parts.username is not None or parts.password is not None:
        raise ValueError("Credentials in URL are not allowed")

    # Validate port if present
    try:
        port = parts.port  # accessing .port validates it's an integer if present
    except ValueError:
        raise ValueError("Invalid port in URL")
    if port is not None and not (1 <= port <= 65535):
        raise ValueError("Port must be between 1 and 65535")

    # Validate hostname
    hostname = parts.hostname  # without brackets for IPv6
    if not _is_valid_host(hostname):
        raise ValueError("Invalid host in URL")

    # Basic path validation: disallow backslashes
    if "\\" in parts.path:
        raise ValueError("URL path must not contain backslashes")

    # Disallow whitespace in netloc
    if any(ch.isspace() for ch in parts.netloc):
        raise ValueError("URL host must not contain whitespace")

    return True


__all__ = ["validate_url"]
