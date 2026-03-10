import re
import ipaddress
from urllib.parse import urlsplit
import urllib.parse

ALLOWED_CDN_DOMAINS = {'cdn.example.com', 'assets.example.net', 'static.example.org'}


def validate_url(url: str) -> bool:
    """
    Validate a URL string.

    Returns:
        True if the URL is valid.

    Raises:
        ValueError: If the URL is invalid.
    """
    if not isinstance(url, str):
        raise ValueError("URL must be a string")

    s = url.strip()
    if not s:
        raise ValueError("URL must be a non-empty string")
    # Reject urls with whitespace/control characters
    if any(c.isspace() for c in s):
        raise ValueError("URL must not contain whitespace characters")

    sr = urlsplit(s)

    if not sr.scheme:
        raise ValueError("URL is missing a scheme")

    scheme = sr.scheme.lower()

    network_schemes = {"http", "https", "ftp", "ftps", "ws", "wss"}
    file_schemes = {"file"}

    if scheme in network_schemes:
        # Must have a hostname
        hostname = sr.hostname
        if not hostname:
            raise ValueError("URL is missing a host")

        if not _is_valid_hostname(hostname):
            raise ValueError("URL host is invalid")

        # Validate port if present
        try:
            port = sr.port
        except ValueError:
            raise ValueError("URL port is invalid")
        if port is not None and not (1 <= port <= 65535):
            raise ValueError("URL port is out of range")

    elif scheme in file_schemes:
        # file:// may have a host (UNC) or be local; must have a path or a netloc
        if not sr.path and not sr.netloc:
            raise ValueError("file URL must include a path or network location")
        # If a hostname is present in file URL, validate it
        if sr.hostname and not _is_valid_hostname(sr.hostname):
            raise ValueError("file URL host is invalid")
    else:
        raise ValueError(f"Unsupported URL scheme: {scheme}")

    return True


def construct_resource_path(res_id: str, res_type: str) -> str:
    """
    Construct a path for a resource organized by type and identifier.

    The path format is: "<res_type>/<res_id>"

    Both res_type and res_id must be non-empty strings without whitespace,
    path separators, or traversal segments.

    Raises:
        ValueError: If either input is invalid.
    """
    type_seg = _validate_path_segment(res_type, "res_type")
    id_seg = _validate_path_segment(res_id, "res_id")
    return f"{type_seg}/{id_seg}"


def create_resource_url(base_cdn_url: str, res_id: str, res_type: str) -> str:
    """
    Create a complete CDN resource URL from base URL, type, and id.

    Validates:
      - base_cdn_url is a valid URL
      - base domain is in ALLOWED_CDN_DOMAINS

    Returns:
      A fully constructed URL string.

    Raises:
      ValueError on any validation failure.
    """
    if not isinstance(base_cdn_url, str):
        raise ValueError("base_cdn_url must be a string")

    # Validate the base URL format first
    validate_url(base_cdn_url)

    sr = urllib.parse.urlsplit(base_cdn_url.strip())
    hostname = sr.hostname
    if not hostname:
        raise ValueError("Base CDN URL is missing a host")

    # Normalize hostname for comparison (IDNA/ASCII, lowercase)
    try:
        ascii_host = hostname.encode("idna").decode("ascii").lower()
    except Exception:
        raise ValueError("Base CDN URL host is invalid")

    if ascii_host not in ALLOWED_CDN_DOMAINS:
        raise ValueError("Base CDN domain is not allowed")

    # Build resource path safely
    resource_path = construct_resource_path(res_id, res_type)

    # Join base path and resource path
    base_path = sr.path or ""
    if base_path.endswith("/"):
        new_path = base_path + resource_path
    elif base_path == "":
        new_path = "/" + resource_path
    else:
        new_path = base_path + "/" + resource_path

    # Reconstruct the URL, preserving query and fragment if present
    return urllib.parse.urlunsplit((sr.scheme, sr.netloc, new_path, sr.query, sr.fragment))


def _validate_path_segment(segment: str, seg_name: str) -> str:
    if not isinstance(segment, str):
        raise ValueError(f"{seg_name} must be a string")
    s = segment.strip()
    if not s:
        raise ValueError(f"{seg_name} must be a non-empty string")
    if any(ch.isspace() for ch in s):
        raise ValueError(f"{seg_name} must not contain whitespace characters")
    if "\x00" in s:
        raise ValueError(f"{seg_name} must not contain NUL characters")
    if "/" in s or "\\" in s:
        raise ValueError(f"{seg_name} must not contain path separators")
    if s in {".", ".."}:
        raise ValueError(f"{seg_name} must not be '.' or '..'")
    return s


def _is_valid_hostname(hostname: str) -> bool:
    """
    Validate a hostname which may be:
    - IPv4 or IPv6 address
    - Domain name (including IDN/Unicode via IDNA)
    - 'localhost'
    """
    if not hostname:
        return False

    # Accept localhost
    if hostname.lower() == "localhost":
        return True

    # Try IP address (IPv4 or IPv6)
    try:
        ipaddress.ip_address(hostname)
        return True
    except ValueError:
        pass

    # Colons are not allowed in regular hostnames (only in IPv6 addressed above)
    if ":" in hostname:
        return False

    # Convert Unicode domains to ASCII using IDNA (punycode)
    try:
        ascii_hostname = hostname.encode("idna").decode("ascii")
    except Exception:
        return False

    if len(ascii_hostname) > 253:
        return False

    labels = ascii_hostname.split(".")
    # Disallow empty labels (no leading/trailing/consecutive dots)
    if any(label == "" for label in labels):
        return False

    label_re = re.compile(r"^[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?$")
    for label in labels:
        if len(label) > 63:
            return False
        if not label_re.match(label):
            return False

    return True
