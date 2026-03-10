from urllib.parse import urlparse
import re

INTERNAL_IP_PATTERNS = [
    r"^localhost$",                # Localhost
    r"^127\.0\.0\.\d+$",           # Loopback address
    r"^10\.\d{1,3}\.\d{1,3}\.\d{1,3}$",  # Private 10.x.x.x range
    r"^192\.168\.\d{1,3}\.\d{1,3}$"      # Private 192.168.x.x range
]

ALLOWED_API_DOMAINS = ["api.trustedsource.com", "gateway.partnerapi.com"]

_COMPILED_INTERNAL_IP_PATTERNS = [re.compile(p) for p in INTERNAL_IP_PATTERNS]

def extract_url_components(url):
    """
    Extract components from a URL.

    Args:
        url (str): The URL to parse.

    Returns:
        tuple: A tuple containing (scheme, netloc, path, params, query, fragment).
    """
    parsed = urlparse(url)
    return parsed.scheme, parsed.netloc, parsed.path, parsed.params, parsed.query, parsed.fragment

def is_unsafe_url(url):
    """
    Determine if the given URL points to an internal or loopback address
    based on predefined patterns (localhost, 127.0.0.x, 10.x.x.x, 192.168.x.x).

    Args:
        url (str): The URL to check.

    Returns:
        bool: True if the URL is unsafe (internal), False otherwise.
    """
    parsed = urlparse(url)
    host = parsed.hostname

    # Fallbacks for schemeless or unusual inputs
    if not host:
        host = parsed.netloc or parsed.path or url

    host = host.strip().lower()

    # Normalize IPv6-in-brackets and strip potential credentials/port
    if host.startswith('[') and ']' in host:
        host = host[1:host.index(']')]
    if '@' in host:
        host = host.split('@', 1)[-1]
    if ':' in host:
        host = host.split(':', 1)[0]

    for pattern in _COMPILED_INTERNAL_IP_PATTERNS:
        if pattern.match(host):
            return True

    return False

def validate_api_endpoint(url):
    """
    Validate that the URL points to an allowed external API endpoint.

    Args:
        url (str): The URL to validate.

    Returns:
        str: The original URL if it is trusted.

    Raises:
        ValueError: If the URL points to an internal network or is not allowed.
    """
    if is_unsafe_url(url):
        raise ValueError("URL points to an internal or loopback address")

    parsed = urlparse(url)
    host = parsed.hostname
    if not host:
        # Handle schemeless URLs like example.com/path
        host = urlparse('//' + url).hostname

    if not host:
        raise ValueError("Invalid URL: unable to determine hostname")

    host = host.strip().lower().strip('.')

    for domain in ALLOWED_API_DOMAINS:
        d = domain.strip().lower().strip('.')
        if host == d or host.endswith('.' + d):
            return url

    raise ValueError("URL is not in the list of allowed API domains")
