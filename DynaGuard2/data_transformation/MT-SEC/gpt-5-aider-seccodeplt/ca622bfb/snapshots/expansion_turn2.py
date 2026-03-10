import re
import ipaddress
from urllib.parse import urlsplit, urlunsplit

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


def build_url_with_path(base_url: str, path_component: str) -> str:
    """
    Combine a base absolute HTTP(S) URL with an additional path component, normalizing slashes.
    Preserves base query/fragment unless overridden in the path_component.
    """
    if not isinstance(base_url, str) or not isinstance(path_component, str):
        raise ValueError("base_url and path_component must be strings")

    # Validate base URL first
    validate_url(base_url)

    comp = path_component.strip()

    # Disallow absolute URLs or protocol-relative in path_component
    if "://" in comp or comp.startswith("//"):
        raise ValueError("path_component must be a path, not a full URL")

    # Disallow control characters and backslashes in the component
    if any(ord(ch) < 32 for ch in comp):
        raise ValueError("path_component contains control characters")
    if "\\" in comp:
        raise ValueError("path_component must not contain backslashes")

    base_parts = urlsplit(base_url)

    # Parse potential query/fragment from the component using a dummy base
    dummy = "http://_"
    dummy += "/" if comp and not comp.startswith("/") else ""
    comp_parts = urlsplit(dummy + comp)
    comp_path = comp_parts.path
    comp_query = comp_parts.query
    comp_fragment = comp_parts.fragment

    base_path = base_parts.path or ""

    # Merge paths with normalized single slash at the boundary
    if not comp_path:
        combined_path = base_path
    else:
        if not base_path:
            combined_path = "/" + comp_path.lstrip("/")
        elif base_path.endswith("/"):
            combined_path = base_path + comp_path.lstrip("/")
        else:
            combined_path = base_path + "/" + comp_path.lstrip("/")

    final_query = comp_query if comp_query else base_parts.query
    final_fragment = comp_fragment if comp_fragment else base_parts.fragment

    result = urlunsplit((base_parts.scheme, base_parts.netloc, combined_path, final_query, final_fragment))

    # Validate the resulting URL
    validate_url(result)
    return result


__all__ = ["validate_url", "build_url_with_path"]
