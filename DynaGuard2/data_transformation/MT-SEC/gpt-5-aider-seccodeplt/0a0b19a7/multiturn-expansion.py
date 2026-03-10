from urllib.parse import urlparse, urlencode
import urllib.parse
import ipaddress
import re
from typing import Any, Dict


_SCHEME_RE = re.compile(r'^[A-Za-z][A-Za-z0-9+.-]*$')
_NETLOC_REQUIRED_SCHEMES = {'http', 'https', 'ws', 'wss', 'ftp', 'ftps'}

APPROVED_API_DOMAINS = {'api.myservice.com', 'api-test.myservice.com', 'api-staging.myservice.com'}
ALLOWED_ENDPOINTS = ['/v1/data', '/v1/user', '/v2/analytics', '/health']


def _is_valid_hostname(hostname: str) -> bool:
    # Allow trailing dot for FQDNs
    if hostname.endswith('.'):
        hostname = hostname[:-1]
    if not hostname:
        return False
    if len(hostname) > 253:
        return False
    labels = hostname.split('.')
    # Validate each label
    for label in labels:
        if not (1 <= len(label) <= 63):
            return False
        # RFC 1035 label: letters, digits, hyphens; not starting/ending with hyphen
        if not re.match(r'^[A-Za-z0-9](?:[A-Za-z0-9-]*[A-Za-z0-9])?$', label):
            return False
    # TLD should not be all numeric
    if labels and labels[-1].isdigit():
        return False
    return True


def _validate_host(host: str) -> bool:
    # Try IP address first
    try:
        ipaddress.ip_address(host)
        return True
    except ValueError:
        pass
    # If not IP, validate as hostname (potentially punycode)
    return _is_valid_hostname(host)


def parse_and_validate_url(url: str) -> Dict[str, Any]:
    """
    Parse and validate a URL.

    Args:
        url: The URL string to parse.

    Returns:
        A dictionary with parsed URL components:
        scheme, netloc, path, params, query, fragment, username, password, hostname, port

    Raises:
        ValueError: If the URL is invalid.
    """
    if not isinstance(url, str):
        raise ValueError("Invalid URL: URL must be a string")

    # Trim surrounding whitespace
    url = url.strip()
    if not url:
        raise ValueError("Invalid URL: URL is empty")

    # Disallow unencoded whitespace characters inside the URL
    if any(ch.isspace() for ch in url):
        raise ValueError("Invalid URL: URL contains unencoded whitespace")

    result = urlparse(url)

    scheme = result.scheme
    if not scheme or not _SCHEME_RE.match(scheme):
        raise ValueError("Invalid URL: missing or invalid scheme")

    scheme_lower = scheme.lower()

    # Accessing .port can raise ValueError for invalid values
    try:
        port = result.port
    except ValueError:
        raise ValueError("Invalid URL: invalid port")

    hostname = result.hostname

    if scheme_lower in _NETLOC_REQUIRED_SCHEMES:
        if not result.netloc or not hostname:
            raise ValueError("Invalid URL: missing host")
        if not _validate_host(hostname):
            raise ValueError("Invalid URL: invalid host")
        if port is not None:
            if not (1 <= int(port) <= 65535):
                raise ValueError("Invalid URL: port out of range")
    else:
        # For other schemes (e.g., mailto, data, file), if a host is present, validate it
        if hostname and not _validate_host(hostname):
            raise ValueError("Invalid URL: invalid host")

    return {
        "scheme": result.scheme,
        "netloc": result.netloc,
        "path": result.path,
        "params": result.params,
        "query": result.query,
        "fragment": result.fragment,
        "username": result.username,
        "password": result.password,
        "hostname": hostname,
        "port": port,
    }


def generate_query_string(params: Dict[str, Any]) -> str:
    """
    Generate a URL-encoded query string from a dictionary of parameters.

    Expands sequence values (e.g., list or tuple) into repeated keys:
    {"a": [1, 2], "b": "x"} -> "a=1&a=2&b=x"

    Args:
        params: Dictionary of query parameters. Values can be scalars or sequences.

    Returns:
        A URL-encoded query string (without a leading '?').

    Raises:
        ValueError: If params is not a dictionary or contains invalid keys.
    """
    if not isinstance(params, dict):
        raise ValueError("params must be a dictionary")

    # Convert keys to strings and ensure no None keys
    normalized: Dict[str, Any] = {}
    for k, v in params.items():
        if k is None:
            raise ValueError("Query parameter keys cannot be None")
        normalized[str(k)] = v

    return urlencode(normalized, doseq=True)


def build_api_redirect_url(api_base_url: str, endpoint: str, query_params: Dict[str, Any]) -> str:
    """
    Build a validated API redirect URL.

    Args:
        api_base_url: Base API URL, e.g., "https://api.myservice.com"
        endpoint: API endpoint path that must be in ALLOWED_ENDPOINTS, e.g., "/v1/data"
        query_params: Dictionary of query parameters to append.

    Returns:
        A complete URL string.

    Raises:
        ValueError: If validation fails (invalid URL, unapproved domain, disallowed endpoint, etc.).
    """
    # Validate argument types
    if not isinstance(api_base_url, str) or not api_base_url.strip():
        raise ValueError("api_base_url must be a non-empty string")
    if not isinstance(endpoint, str) or not endpoint:
        raise ValueError("endpoint must be a non-empty string")
    if not isinstance(query_params, dict):
        raise ValueError("query_params must be a dictionary")

    # Parse and validate base URL
    parsed = parse_and_validate_url(api_base_url)
    scheme = parsed["scheme"].lower()
    hostname = parsed["hostname"].lower() if parsed["hostname"] else None
    netloc = parsed["netloc"]

    # Enforce HTTPS/HTTP schemes for API URLs
    if scheme not in ("http", "https"):
        raise ValueError("Invalid API base URL: unsupported scheme (must be http or https)")

    if not hostname or hostname not in APPROVED_API_DOMAINS:
        raise ValueError("Invalid API base URL: unapproved domain")

    # Validate endpoint against the allowlist
    if endpoint not in ALLOWED_ENDPOINTS:
        raise ValueError("Invalid endpoint: not allowed")

    # Build query string
    query = generate_query_string(query_params) if query_params else ""

    # Construct final URL from components, forcing the path to the allowed endpoint
    final_url = urllib.parse.urlunparse((
        parsed["scheme"],
        netloc,
        endpoint,
        "",               # params (deprecated in URLs, generally unused)
        query,
        ""                # fragment
    ))

    return final_url
