import re
import ipaddress
from urllib.parse import urlparse, urlencode
from collections.abc import Mapping
import urllib.parse

APPROVED_API_DOMAINS = {'api.myservice.com', 'api-test.myservice.com', 'api-staging.myservice.com'}
ALLOWED_ENDPOINTS = ['/v1/data', '/v1/user', '/v2/analytics', '/health']


_DOMAIN_REGEX = re.compile(
    r"^(?=.{1,253}$)(?:[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?\.)+[A-Za-z]{2,63}$"
)


def _is_valid_hostname(hostname: str) -> bool:
    if not hostname:
        return False
    if hostname.lower() == "localhost":
        return True
    # Try IP (v4 or v6)
    try:
        ipaddress.ip_address(hostname)
        return True
    except ValueError:
        pass
    # Domain/FQDN
    return _DOMAIN_REGEX.match(hostname) is not None


def validate_and_parse_url(url: str) -> dict:
    """
    Validate the given URL and return its components.

    Args:
        url: The URL string to validate and parse.

    Returns:
        A dict with keys: scheme, domain, port, path, query, fragment, username, password.

    Raises:
        ValueError: If the URL is invalid.
    """
    if not isinstance(url, str):
        raise ValueError("URL must be a string")

    candidate = url.strip()
    if not candidate:
        raise ValueError("URL must not be empty")

    parsed = urlparse(candidate)

    # Basic structural checks
    if not parsed.scheme:
        raise ValueError("URL must include a scheme (e.g., 'https')")
    if not parsed.netloc:
        raise ValueError("URL must include a network location (domain)")

    hostname = parsed.hostname  # lowercased, without brackets for IPv6
    if not _is_valid_hostname(hostname or ""):
        raise ValueError("URL has an invalid domain/host")

    # Validate port (may raise ValueError if invalid)
    try:
        port = parsed.port
    except ValueError:
        raise ValueError("URL has an invalid port")

    # Normalize path: default to "/" for network URLs
    path = parsed.path if parsed.path else "/"

    return {
        "scheme": parsed.scheme,
        "domain": hostname,
        "port": port,
        "path": path,
        "query": parsed.query,
        "fragment": parsed.fragment,
        "username": parsed.username,
        "password": parsed.password,
    }


def build_query_string(params: Mapping) -> str:
    """
    Build a URL-encoded query string from a mapping of parameters.

    Behavior:
    - Skips keys with value None.
    - Expands list/tuple/set values into repeated keys (?a=1&a=2) via doseq=True.
    - Converts booleans to 'true'/'false' (lowercase).
    - Converts other values to strings.
    - Returns an empty string if no encodable parameters remain.

    Args:
        params: A dict-like mapping of query parameters.

    Returns:
        A URL-encoded query string (without leading '?').

    Raises:
        ValueError: If params is not a mapping.
    """
    if not isinstance(params, Mapping):
        raise ValueError("params must be a mapping (e.g., dict)")

    normalized: dict = {}

    for key, value in params.items():
        if value is None:
            continue

        k = str(key)

        # Handle sequences by expanding into multiple values
        if isinstance(value, (list, tuple, set, frozenset)):
            seq = []
            for item in value:
                if item is None:
                    continue
                if isinstance(item, bool):
                    seq.append("true" if item else "false")
                else:
                    seq.append(str(item))
            if seq:
                normalized[k] = seq
        else:
            if isinstance(value, bool):
                normalized[k] = "true" if value else "false"
            else:
                normalized[k] = str(value)

    if not normalized:
        return ""

    return urlencode(normalized, doseq=True)


def create_api_redirect_url(base_url_api: str, api_endpoint: str, opts: Mapping) -> str:
    """
    Create a fully-formed API redirect URL with validation.

    Args:
        base_url_api: Base API URL (must include scheme and approved domain).
        api_endpoint: API endpoint path that must be in ALLOWED_ENDPOINTS.
        opts: Mapping of query parameters to append.

    Returns:
        The complete redirect URL as a string.

    Raises:
        ValueError: If inputs are invalid, base domain not approved, or endpoint not allowed.
    """
    if not isinstance(base_url_api, str):
        raise ValueError("base_url_api must be a string")
    if not isinstance(api_endpoint, str):
        raise ValueError("api_endpoint must be a string")
    if not isinstance(opts, Mapping):
        raise ValueError("opts must be a mapping (e.g., dict)")

    base = validate_and_parse_url(base_url_api)

    domain = base["domain"]
    if domain not in APPROVED_API_DOMAINS:
        raise ValueError("Base API domain is not approved")

    if api_endpoint not in ALLOWED_ENDPOINTS:
        raise ValueError("API endpoint is not allowed")

    # Build query string
    query = build_query_string(opts)

    # Assemble netloc (include port if present)
    netloc = domain if base["port"] is None else f"{domain}:{base['port']}"

    # Construct the final URL
    final_url = urllib.parse.urlunparse((
        base["scheme"],
        netloc,
        api_endpoint,
        "",
        query,
        "",
    ))

    return final_url
