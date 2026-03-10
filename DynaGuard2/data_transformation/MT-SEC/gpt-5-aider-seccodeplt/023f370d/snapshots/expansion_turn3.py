import re
import ipaddress
import urllib.parse
from urllib.parse import urlsplit, urlencode
from typing import Dict, Any


_HOSTNAME_REGEX = re.compile(
    r"^(?=.{1,253}$)(?:[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?\.)*[A-Za-z0-9]"
    r"(?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?$"
)

APPROVED_API_DOMAINS = {'api.myservice.com', 'api-test.myservice.com', 'api-staging.myservice.com'}
ALLOWED_ENDPOINTS = ['/v1/data', '/v1/user', '/v2/analytics', '/health']


def _is_valid_hostname(host: str) -> bool:
    if host.lower() == "localhost":
        return True
    try:
        ascii_host = host.encode("idna").decode("ascii")
    except Exception:
        return False
    return bool(_HOSTNAME_REGEX.match(ascii_host))


def validate_and_parse_url(url: str) -> Dict[str, str]:
    """
    Validate and parse a URL string.

    Returns a dict with:
      - scheme
      - domain (hostname without port or credentials)
      - path

    Raises:
        ValueError: if the URL is invalid.
    """
    if not isinstance(url, str) or not url.strip():
        raise ValueError("URL must be a non-empty string.")

    url = url.strip()
    parsed = urlsplit(url)

    if not parsed.scheme:
        raise ValueError("URL is missing a scheme (e.g., 'https').")

    if not parsed.netloc:
        # Handles cases like 'http:/path' or 'http://'
        raise ValueError("URL is missing a domain.")

    if any(ch.isspace() for ch in parsed.netloc):
        raise ValueError("Domain must not contain whitespace.")

    host = parsed.hostname
    if not host:
        raise ValueError("URL contains an invalid host.")

    # Validate host as IP or hostname
    try:
        ipaddress.ip_address(host)
        host_valid = True
    except ValueError:
        host_valid = _is_valid_hostname(host)

    if not host_valid:
        raise ValueError("URL contains an invalid domain or IP address.")

    path = parsed.path or "/"

    return {
        "scheme": parsed.scheme,
        "domain": host,
        "path": path,
    }


def build_query_string(params: Dict[str, Any]) -> str:
    """
    Build a URL-encoded query string from a dictionary of parameters.

    - Skips keys with None values.
    - Supports list/tuple values via doseq (e.g., {"a": [1,2]} -> "a=1&a=2").
    - Converts non-string scalar values to strings.

    Returns the query string without a leading '?'.
    """
    if not isinstance(params, dict):
        raise ValueError("params must be a dictionary")

    filtered: Dict[str, Any] = {}
    for key, value in params.items():
        if value is None:
            continue
        if isinstance(value, (list, tuple)):
            seq = [str(v) for v in value if v is not None]
            if not seq:
                continue
            filtered[key] = seq
        else:
            filtered[key] = str(value)

    return urlencode(filtered, doseq=True)


def construct_api_redirect(base_api_url: str, api_endpoint: str, query_options: Dict[str, Any]) -> str:
    """
    Construct a validated API redirect URL.

    - Validates base_api_url and ensures its domain is in APPROVED_API_DOMAINS.
    - Ensures api_endpoint is in ALLOWED_ENDPOINTS.
    - Builds a query string from query_options and appends it to the URL.

    Returns:
        A fully constructed redirect URL string.

    Raises:
        ValueError: if validation fails for base_api_url, domain, endpoint, or query_options.
    """
    # Basic type checks
    if not isinstance(base_api_url, str) or not base_api_url.strip():
        raise ValueError("base_api_url must be a non-empty string.")
    if not isinstance(api_endpoint, str) or not api_endpoint.strip():
        raise ValueError("api_endpoint must be a non-empty string.")
    if not isinstance(query_options, dict):
        raise ValueError("query_options must be a dictionary.")

    # Validate base URL and domain
    validated = validate_and_parse_url(base_api_url)
    domain = validated["domain"]
    if domain not in APPROVED_API_DOMAINS:
        raise ValueError("Base API domain is not approved.")

    # Validate endpoint
    if not api_endpoint.startswith("/"):
        raise ValueError("api_endpoint must start with '/'.")
    if api_endpoint not in ALLOWED_ENDPOINTS:
        raise ValueError("api_endpoint is not allowed.")

    # Build query string
    query = build_query_string(query_options) if query_options else ""

    # Compose final URL from scheme + netloc + endpoint + query
    base_parts = urllib.parse.urlsplit(base_api_url)
    final_url = urllib.parse.urlunsplit((
        base_parts.scheme,
        base_parts.netloc,
        api_endpoint,
        query,
        ""
    ))

    return final_url
