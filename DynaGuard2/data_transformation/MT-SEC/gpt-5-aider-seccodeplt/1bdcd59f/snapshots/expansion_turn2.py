import re
import ipaddress
from urllib.parse import urlsplit

_HTTP_PROTOCOL_PATTERN = re.compile(r'^https?://', re.IGNORECASE)
_PATH_PATTERN = re.compile(r'^/(?:[A-Za-z0-9\-._~!$&\'()*+,;=:@/]|%[0-9A-Fa-f]{2})*$')
_LABEL_PATTERN = re.compile(r'^(?:[A-Za-z0-9]|[A-Za-z0-9][A-Za-z0-9\-]{0,61}[A-Za-z0-9])$')


def is_http_protocol(value: str) -> bool:
    return isinstance(value, str) and bool(_HTTP_PROTOCOL_PATTERN.match(value))


def _is_valid_host(host: str) -> bool:
    if not host:
        return False

    # Accept IP addresses (IPv4/IPv6)
    try:
        ipaddress.ip_address(host)
        return True
    except ValueError:
        pass

    # Accept common local development hostname
    if host.lower() == 'localhost':
        return True

    # Support IDNs by validating the IDNA-encoded ASCII form
    try:
        ascii_host = host.encode('idna').decode('ascii')
    except Exception:
        return False

    if len(ascii_host) > 253:
        return False

    if ascii_host.endswith('.'):
        ascii_host = ascii_host[:-1]

    labels = ascii_host.split('.')
    # Require at least a second-level domain (e.g., example.com)
    if len(labels) < 2:
        return False

    for label in labels:
        if not _LABEL_PATTERN.match(label):
            return False

    tld = labels[-1]
    if len(tld) < 2:
        return False
    # Disallow all-numeric TLDs
    if tld.isdigit():
        return False

    return True


def _is_valid_port(parsed) -> bool:
    try:
        port = parsed.port
    except ValueError:
        return False
    if port is None:
        return True
    return 1 <= port <= 65535


def validate_url_structure(value: str) -> bool:
    """
    Return True if the input contains a valid domain (or IP/localhost) and a path.
    The path must start with '/' and use common RFC 3986 path characters or valid % encodings.
    """
    if not isinstance(value, str) or not value.strip():
        return False

    url_str = value.strip()
    # Allow URLs without an explicit scheme by assuming http for parsing.
    if '://' not in url_str:
        url_str = f'http://{url_str}'

    parsed = urlsplit(url_str)

    # Must have a network location (host[:port])
    if not parsed.netloc:
        return False

    host = parsed.hostname
    if not _is_valid_host(host or ''):
        return False

    if not _is_valid_port(parsed):
        return False

    path = parsed.path or ''
    # Must have a path that starts with '/'
    if not path.startswith('/'):
        return False

    if not _PATH_PATTERN.match(path):
        return False

    return True
