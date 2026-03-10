import re
import ipaddress
from urllib.parse import urlsplit, urlunsplit, urljoin, quote


_ALLOWED_SCHEMES = {"http", "https", "ws", "wss", "ftp", "ftps"}


def _is_valid_hostname(hostname: str) -> bool:
    # Accept IP addresses (IPv4/IPv6)
    try:
        ipaddress.ip_address(hostname)
        return True
    except ValueError:
        pass

    # Validate IDNA-encodable hostnames
    try:
        ascii_host = hostname.encode("idna").decode("ascii")
    except UnicodeError:
        return False

    if len(ascii_host) > 253:
        return False

    # Allow trailing dot in hostname representation for FQDNs
    if ascii_host.endswith("."):
        ascii_host = ascii_host[:-1]

    labels = ascii_host.split(".")
    label_re = re.compile(r"^[A-Za-z0-9-]{1,63}$")

    for label in labels:
        if not label:
            return False
        if not label_re.match(label):
            return False
        if label[0] == "-" or label[-1] == "-":
            return False

    return True


def validate_url(url: str) -> bool:
    """
    Validate that the given URL is well-formed.
    Returns True if valid; otherwise raises ValueError.
    """
    if not isinstance(url, str):
        raise ValueError("URL must be a string")

    candidate = url.strip()
    if not candidate:
        raise ValueError("URL is empty")

    if re.search(r"\s", candidate):
        raise ValueError("URL must not contain whitespace characters")

    parsed = urlsplit(candidate)

    if not parsed.scheme:
        raise ValueError("URL must include a scheme (e.g., http, https)")
    if parsed.scheme.lower() not in _ALLOWED_SCHEMES:
        raise ValueError(f"Unsupported URL scheme: {parsed.scheme}")

    if not parsed.netloc:
        raise ValueError("URL must include a network location (host)")

    hostname = parsed.hostname
    if not hostname:
        raise ValueError("URL must include a valid hostname")

    if not _is_valid_hostname(hostname):
        raise ValueError("Invalid hostname in URL")

    # Validate port (if present)
    try:
        port = parsed.port
    except ValueError:
        raise ValueError("Invalid port in URL")
    if port is not None and not (1 <= port <= 65535):
        raise ValueError("Port out of valid range (1-65535)")

    return True


def concatenate_url_path(base_url: str, path: str) -> str:
    """
    Concatenate a URL path to a base URL and return the full URL.

    Rules:
    - base_url must be a valid URL (validated via validate_url).
    - path must be a string path (no scheme or netloc). It may include query/fragment.
    - If path is empty or whitespace, base_url is returned unchanged.
    - If path is absolute (starts with '/'), it is resolved from the domain root.
    - Query/fragment in path override those in base_url; otherwise base URL's are preserved.
    - Path components are percent-encoded as needed (without double-encoding).
    """
    if not isinstance(base_url, str):
        raise ValueError("base_url must be a string")
    if not isinstance(path, str):
        raise ValueError("path must be a string")

    base = base_url.strip()
    segment = path.strip()

    # Validate base URL
    validate_url(base)

    if segment == "":
        return base

    # Disallow full URLs in the path argument
    path_parts = urlsplit(segment)
    if path_parts.scheme or path_parts.netloc:
        raise ValueError("path must be a URL path, not a full URL")

    # Prepare base URL with trailing slash and without query/fragment for joining
    base_parts = urlsplit(base)
    base_path = base_parts.path if base_parts.path else "/"
    if not base_path.endswith("/"):
        base_path = base_path + "/"
    base_for_join = urlunsplit((base_parts.scheme, base_parts.netloc, base_path, "", ""))

    # Percent-encode the path portion appropriately (preserve '/' and existing encodings)
    encoded_path = quote(path_parts.path, safe="/%:@!$&'()*+,;=~.-")

    joined = urljoin(base_for_join, encoded_path)
    joined_parts = urlsplit(joined)

    # Preserve base query/fragment unless overridden by the provided path
    final_query = path_parts.query if path_parts.query else base_parts.query
    final_fragment = path_parts.fragment if path_parts.fragment else base_parts.fragment

    full_url = urlunsplit(
        (joined_parts.scheme, joined_parts.netloc, joined_parts.path, final_query, final_fragment)
    )

    # Final validation
    validate_url(full_url)

    return full_url
