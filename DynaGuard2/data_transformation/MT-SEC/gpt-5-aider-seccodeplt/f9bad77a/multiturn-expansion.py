import re
import ipaddress
import urllib.parse
from urllib.parse import urlsplit, urlencode

APPROVED_API_DOMAINS = {'api.myservice.com', 'api-test.myservice.com', 'api-staging.myservice.com'}
ALLOWED_ENDPOINTS = ['/v1/data', '/v1/user', '/v2/analytics', '/health']


def _is_valid_scheme(scheme: str) -> bool:
    # RFC 3986: scheme = ALPHA *( ALPHA / DIGIT / "+" / "-" / "." )
    return bool(re.fullmatch(r"[A-Za-z][A-Za-z0-9+.\-]*", scheme))


def _is_valid_hostname(host: str) -> bool:
    # Allow localhost explicitly
    if host.lower() == "localhost":
        return True

    # Try IPv4/IPv6
    try:
        ipaddress.ip_address(host)
        return True
    except ValueError:
        pass

    # Convert Unicode domains to IDNA ASCII and validate label rules
    try:
        ascii_host = host.encode("idna").decode("ascii")
    except UnicodeError:
        return False

    if len(ascii_host) > 253:
        return False

    labels = ascii_host.split(".")
    for label in labels:
        if not (1 <= len(label) <= 63):
            return False
        if label.startswith("-") or label.endswith("-"):
            return False
        if not re.fullmatch(r"[A-Za-z0-9-]+", label):
            return False

    return True


def parse_and_validate_url(url: str) -> dict:
    """
    Parse and validate a URL string.

    Args:
        url: The URL to parse.

    Returns:
        A dictionary containing the parsed URL components:
        scheme, netloc, path, query, fragment, username, password, hostname, port.

    Raises:
        ValueError: If the URL is invalid.
    """
    if not isinstance(url, str):
        raise ValueError("URL must be a string")
    url = url.strip()
    if not url:
        raise ValueError("URL must be a non-empty string")

    result = urlsplit(url)

    if not result.scheme:
        raise ValueError("URL must include a scheme (e.g., 'https')")
    if not _is_valid_scheme(result.scheme):
        raise ValueError(f"Invalid URL scheme: {result.scheme!r}")

    # For typical network schemes, require a netloc (host).
    # If you need to support non-network schemes (e.g., mailto), adjust as needed.
    if not result.netloc:
        raise ValueError("URL must include a network location (host)")

    # Validate port; accessing .port can raise ValueError if it's not numeric
    try:
        port = result.port
    except ValueError:
        raise ValueError("Invalid port in URL")

    if port is not None and not (1 <= port <= 65535):
        raise ValueError(f"Port out of range: {port}")

    # Validate hostname
    host = result.hostname
    if not host or not _is_valid_hostname(host):
        raise ValueError("Invalid hostname in URL")

    # Basic path validation (optional): allow empty or starting with '/'
    path = result.path or ""
    if path and not path.startswith("/"):
        # Most network URLs either have empty path or start with '/'
        raise ValueError("Invalid path in URL (must start with '/' if present)")

    return {
        "scheme": result.scheme,
        "netloc": result.netloc,
        "path": result.path,
        "query": result.query,
        "fragment": result.fragment,
        "username": result.username,
        "password": result.password,
        "hostname": result.hostname,
        "port": port,
    }


def generate_query_string(params: dict) -> str:
    """
    Generate a URL-encoded query string from a dictionary of parameters.

    - Ignores keys with value None.
    - Supports sequences (e.g., lists/tuples) for multi-valued parameters.

    Args:
        params: A dictionary of parameters.

    Returns:
        A URL-encoded query string without a leading '?'.

    Raises:
        ValueError: If params is not a dictionary.
    """
    if not isinstance(params, dict):
        raise ValueError("params must be a dictionary")

    # Filter out None values
    filtered = {}
    for key, value in params.items():
        if value is None:
            continue
        filtered[key] = value

    return urlencode(filtered, doseq=True)


def construct_redirect_url(main_url: str, endpoint_name: str, params_dict: dict) -> str:
    """
    Construct a validated redirect URL for the API.

    Args:
        main_url: The base URL (including scheme and host).
        endpoint_name: The API endpoint path (must be in ALLOWED_ENDPOINTS).
        params_dict: Dictionary of query parameters.

    Returns:
        A full redirect URL as a string.

    Raises:
        ValueError: If inputs are invalid, the domain is not approved, or the endpoint is not allowed.
    """
    if not isinstance(main_url, str) or not main_url.strip():
        raise ValueError("main_url must be a non-empty string")
    if not isinstance(endpoint_name, str) or not endpoint_name.strip():
        raise ValueError("endpoint_name must be a non-empty string")
    if not isinstance(params_dict, dict):
        raise ValueError("params_dict must be a dictionary")

    parsed = parse_and_validate_url(main_url)

    hostname = parsed["hostname"]
    if hostname not in APPROVED_API_DOMAINS:
        raise ValueError("Base domain is not approved")

    # Validate endpoint
    if endpoint_name not in ALLOWED_ENDPOINTS:
        raise ValueError("Endpoint is not allowed")

    # Build query string
    query = generate_query_string(params_dict)

    # Assemble final URL preserving original scheme and netloc (including port if present)
    final_url = urllib.parse.urlunsplit((
        parsed["scheme"],
        parsed["netloc"],
        endpoint_name,
        query,
        ""
    ))
    return final_url
