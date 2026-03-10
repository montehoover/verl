import re
import ipaddress
import urllib.parse
from urllib.parse import urlsplit, urlencode
from typing import Dict, Any


APPROVED_API_DOMAINS = {'api.myservice.com', 'api-test.myservice.com', 'api-staging.myservice.com'}
ALLOWED_ENDPOINTS = ['/v1/data', '/v1/user', '/v2/analytics', '/health']

_SCHEME_RE = re.compile(r"^[A-Za-z][A-Za-z0-9+.-]*$")


def _is_valid_scheme(scheme: str) -> bool:
    if not scheme or not isinstance(scheme, str):
        return False
    return _SCHEME_RE.match(scheme) is not None


def _is_valid_domain(host: str) -> bool:
    if not host or not isinstance(host, str):
        return False

    # Try IP literal (both IPv4 and IPv6)
    try:
        ipaddress.ip_address(host)
        return True
    except ValueError:
        pass

    # Validate IDN/hostname by converting to IDNA (punycode) then checking labels
    try:
        host_idna = host.encode("idna").decode("ascii")
    except UnicodeError:
        return False

    if len(host_idna) > 253:
        return False

    labels = host_idna.split(".")
    for label in labels:
        if not label or len(label) > 63:
            return False
        # Must start/end alphanumeric; can contain hyphens inside
        if not re.fullmatch(r"[A-Za-z0-9](?:[A-Za-z0-9-]*[A-Za-z0-9])?", label):
            return False

    return True


def parse_and_validate_url(url: str) -> Dict[str, str]:
    """
    Parse and validate a URL.

    Args:
        url: The URL string to parse.

    Returns:
        A dict with at least the keys: 'scheme', 'domain', and 'path'.

    Raises:
        ValueError: If the URL is invalid.
    """
    if not isinstance(url, str):
        raise ValueError("URL must be a string")

    url = url.strip()
    if not url:
        raise ValueError("URL must not be empty")

    parsed = urlsplit(url)

    # Validate scheme
    if not _is_valid_scheme(parsed.scheme):
        raise ValueError("Invalid or missing URL scheme")

    # Extract hostname (domain) and validate
    host = parsed.hostname
    if host is None or not _is_valid_domain(host):
        raise ValueError("Invalid or missing URL domain/host")

    # Validate port if present (urlsplit.port may raise ValueError for bad ports)
    try:
        _ = parsed.port  # Accessing triggers validation
    except ValueError:
        raise ValueError("Invalid port in URL")

    # Build result dictionary
    result = {
        "scheme": parsed.scheme.lower(),
        "domain": host,
        "path": parsed.path or "",
    }

    return result


def _stringify_value(value: Any) -> str:
    """Convert a value to a string suitable for query parameters."""
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def build_query_string(params: Dict[str, Any]) -> str:
    """
    Build a URL-encoded query string from a dictionary of parameters.

    - Skips parameters with a value of None.
    - If a value is a list/tuple, includes the key for each non-None item.
    - Booleans are rendered as 'true'/'false'.

    Args:
        params: A dictionary of query parameters.

    Returns:
        A URL-encoded query string (without a leading '?').

    Raises:
        ValueError: If params is not a dictionary.
    """
    if not isinstance(params, dict):
        raise ValueError("params must be a dictionary")

    pairs = []
    for key, value in params.items():
        if value is None:
            continue
        k = str(key)
        if isinstance(value, (list, tuple)):
            seq = [item for item in value if item is not None]
            if not seq:
                continue
            for item in seq:
                pairs.append((k, _stringify_value(item)))
        else:
            pairs.append((k, _stringify_value(value)))

    return urlencode(pairs, doseq=True)


def _is_allowed_endpoint(path: str) -> bool:
    if not isinstance(path, str) or not path:
        return False
    norm = path if path.startswith("/") else "/" + path
    for allowed in ALLOWED_ENDPOINTS:
        if norm == allowed or norm.startswith(allowed + "/"):
            return True
    return False


def generate_api_redirect(base_url: str, api_path: str, params: Dict[str, Any]) -> str:
    """
    Generate a validated API redirect URL.

    - base_url must have an approved domain.
    - api_path must be an allowed endpoint (exact match or subpath).
    - params is URL-encoded and appended as the query string.

    Raises:
        ValueError if validation fails.
    """
    if not isinstance(base_url, str) or not base_url.strip():
        raise ValueError("base_url must be a non-empty string")
    if not isinstance(api_path, str) or not api_path.strip():
        raise ValueError("api_path must be a non-empty string")
    if not isinstance(params, dict):
        raise ValueError("params must be a dictionary")

    # Validate base URL and domain
    base_info = parse_and_validate_url(base_url)
    domain = base_info["domain"].lower()
    if domain not in APPROVED_API_DOMAINS:
        raise ValueError("Base domain is not approved")

    base_parts = urlsplit(base_url)
    if base_parts.username or base_parts.password:
        raise ValueError("User info in base URL is not allowed")

    # Normalize endpoint path and validate against allowed endpoints
    api_path = api_path if api_path.startswith("/") else "/" + api_path
    if not _is_allowed_endpoint(api_path):
        raise ValueError("API endpoint is not allowed")

    # Build base origin (preserve explicit port if present)
    if base_parts.port:
        netloc = f"{domain}:{base_parts.port}"
    else:
        netloc = domain
    origin = f"{base_parts.scheme}://{netloc}"

    # Build full path
    full_path = urllib.parse.urljoin(origin, api_path)

    # Build query string
    query = build_query_string(params)
    return f"{full_path}?{query}" if query else full_path
