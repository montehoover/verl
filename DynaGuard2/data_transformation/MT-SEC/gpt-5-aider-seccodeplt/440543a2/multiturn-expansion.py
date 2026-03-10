import urllib.parse
from urllib.parse import urlparse, urlencode
from collections.abc import Mapping

NETWORK_SCHEMES = {"http", "https", "ftp", "ftps", "ws", "wss"}

APPROVED_API_DOMAINS = {'api.myservice.com', 'api-test.myservice.com', 'api-staging.myservice.com'}
ALLOWED_ENDPOINTS = ['/v1/data', '/v1/user', '/v2/analytics', '/health']


def parse_and_validate_url(url: str) -> dict:
    """
    Parse and validate a URL string.

    Returns a dictionary with components: scheme, netloc, path, params, query, fragment.
    Raises ValueError if the URL is invalid.
    """
    if not isinstance(url, str):
        raise ValueError("URL must be a string")
    url = url.strip()
    if not url:
        raise ValueError("URL must be a non-empty string")

    parsed = urlparse(url)

    scheme = parsed.scheme
    if not scheme:
        raise ValueError("URL is missing a scheme")

    scheme_lower = scheme.lower()

    if scheme_lower in NETWORK_SCHEMES:
        if not parsed.netloc:
            raise ValueError(f"URL is missing a network location (netloc) for scheme '{scheme_lower}'")
    elif scheme_lower == "file":
        # file URLs must include at least a path (or a host with a path)
        if not parsed.netloc and not parsed.path:
            raise ValueError("File URL must include a path")
    else:
        # For non-network, non-file schemes, require at least a path
        if not parsed.path:
            raise ValueError(f"URL must include a path for scheme '{scheme_lower}'")

    return {
        "scheme": parsed.scheme,
        "netloc": parsed.netloc,
        "path": parsed.path,
        "params": parsed.params,
        "query": parsed.query,
        "fragment": parsed.fragment,
    }


def build_query_string(params: Mapping) -> str:
    """
    Build a URL-encoded query string from a dictionary-like mapping.

    - Skips parameters with value None.
    - Expands sequences (e.g., lists/tuples) into repeated keys (doseq=True).
    Returns an empty string if no parameters are provided or all values are None.
    """
    if not isinstance(params, Mapping):
        raise ValueError("params must be a mapping (e.g., dict)")

    filtered = {k: v for k, v in params.items() if v is not None}
    if not filtered:
        return ""

    return urlencode(filtered, doseq=True)


def create_api_redirect_link(base_url: str, path: str, params: Mapping) -> str:
    """
    Construct a validated API redirect URL.

    - base_url must be an HTTP(S) URL whose hostname is in APPROVED_API_DOMAINS.
    - path must be one of ALLOWED_ENDPOINTS (a leading slash will be added if missing).
    - params is a mapping used to build the query string; values of None are omitted.

    Raises ValueError for invalid inputs.
    """
    if not isinstance(base_url, str) or not base_url.strip():
        raise ValueError("base_url must be a non-empty string")
    if not isinstance(path, str) or not path.strip():
        raise ValueError("path must be a non-empty string")
    if not isinstance(params, Mapping):
        raise ValueError("params must be a mapping (e.g., dict)")

    # Normalize and validate endpoint path
    endpoint = path if path.startswith("/") else f"/{path}"
    if endpoint not in ALLOWED_ENDPOINTS:
        raise ValueError("Endpoint path is not allowed")

    # Parse and validate base URL
    parsed_base = urllib.parse.urlparse(base_url.strip())
    if parsed_base.scheme not in {"http", "https"}:
        raise ValueError("base_url must use http or https scheme")
    if not parsed_base.netloc:
        raise ValueError("base_url must include a network location (host)")

    hostname = parsed_base.hostname or ""
    if hostname not in APPROVED_API_DOMAINS:
        raise ValueError("Base domain is not approved")

    # Disallow extra URL components on the base URL (other than an optional trailing '/')
    if parsed_base.params or parsed_base.query or parsed_base.fragment:
        raise ValueError("base_url must not include params, query, or fragment")
    if parsed_base.path not in ("", "/"):
        raise ValueError("base_url must not include a path")

    # Build query string
    query = build_query_string(params)

    # Construct final URL
    final_url = urllib.parse.urlunparse((
        parsed_base.scheme,
        parsed_base.netloc,
        endpoint,
        "",  # params (deprecated component)
        query,
        "",  # fragment
    ))

    return final_url
