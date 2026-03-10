import re
import ipaddress
from urllib.parse import urlsplit


_SCHEME_RE = re.compile(r'^[A-Za-z][A-Za-z0-9+\-.]*$')
_NETWORK_SCHEMES = {'http', 'https', 'ftp', 'ftps', 'ws', 'wss'}


def _validate_host(host: str) -> None:
    """
    Validate a host component. Accepts IPv4, IPv6, or DNS names (including Unicode, via IDNA).
    Raises ValueError if invalid.
    """
    # Try IP first
    try:
        ipaddress.ip_address(host)
        return
    except ValueError:
        pass  # Not an IP, continue as a hostname

    # IDNA-encode labels and validate ASCII form rules
    labels = host.split('.')
    # Allow a trailing dot (FQDN) -> results in last empty label
    if labels and labels[-1] == '':
        labels = labels[:-1]

    if not labels:
        raise ValueError("Host is empty")

    try:
        ascii_labels = [lbl.encode('idna').decode('ascii') for lbl in labels]
    except UnicodeError as e:
        raise ValueError(f"Host contains invalid Unicode: {e}") from None

    for lbl in ascii_labels:
        if not lbl:
            raise ValueError("Hostname contains empty label")
        if len(lbl) > 63:
            raise ValueError("Hostname label exceeds 63 characters")
        if not re.fullmatch(r'[A-Za-z0-9-]+', lbl):
            raise ValueError("Hostname label contains invalid characters")
        if lbl.startswith('-') or lbl.endswith('-'):
            raise ValueError("Hostname label must not start or end with hyphen")


def validate_url(url: str) -> bool:
    """
    Validate a URL.
    Returns True if valid. Raises ValueError if invalid.

    Rules:
    - Scheme must exist and match RFC 3986 scheme pattern.
    - For network schemes (http, https, ftp, ftps, ws, wss):
        * Host must be present and valid.
        * Port, if present, must be 0..65535.
    - file: requires a path or UNC host.
    - mailto: requires a non-empty path (address part).
    - data: requires a comma separating mediatype and data.
    - Other schemes: require at least a non-empty netloc or path.
    """
    if not isinstance(url, str):
        raise ValueError("URL must be a string")

    s = url.strip()
    if not s:
        raise ValueError("URL is empty")

    if any(ch.isspace() for ch in s):
        # Spaces must be percent-encoded in URLs
        raise ValueError("URL contains whitespace; encode spaces as %20")

    try:
        parts = urlsplit(s)
    except Exception as e:
        raise ValueError(f"Invalid URL: {e}") from None

    scheme = parts.scheme
    if not scheme or not _SCHEME_RE.match(scheme):
        raise ValueError("URL scheme is missing or invalid")

    scheme_l = scheme.lower()

    if scheme_l in _NETWORK_SCHEMES:
        if not parts.netloc:
            raise ValueError("URL host is missing")
        host = parts.hostname
        if not host:
            raise ValueError("URL host is missing")
        _validate_host(host)

        # Validate port range if present
        try:
            port = parts.port
        except ValueError:
            # Raised if port isn't numeric
            raise ValueError("URL port is not numeric") from None

        if port is not None and not (0 <= port <= 65535):
            raise ValueError("URL port is out of range")

        return True

    if scheme_l == 'file':
        # file:///path or file://host/path
        if not parts.path and not parts.netloc:
            raise ValueError("file URL must include a path or UNC host")
        return True

    if scheme_l == 'mailto':
        if not parts.path:
            raise ValueError("mailto URL must include an address")
        return True

    if scheme_l == 'data':
        # Minimal check: data:[<mediatype>][;base64],<data>
        if ',' not in parts.path:
            raise ValueError("data URL must contain a comma separating metadata and data")
        return True

    # Generic fallback for other schemes:
    if parts.netloc or parts.path:
        return True

    raise ValueError("URL missing required components")
