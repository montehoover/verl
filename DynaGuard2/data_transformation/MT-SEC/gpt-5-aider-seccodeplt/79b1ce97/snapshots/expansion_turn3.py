import urllib.parse
from urllib.parse import urlparse, urlencode

APPROVED_API_DOMAINS = {'api.myservice.com', 'api-test.myservice.com', 'api-staging.myservice.com'}
ALLOWED_ENDPOINTS = ['/v1/data', '/v1/user', '/v2/analytics', '/health']


def parse_and_validate_url(url: str) -> dict:
    """
    Parse and validate a URL string.

    Args:
        url (str): The URL to parse.

    Returns:
        dict: A dictionary with keys 'scheme', 'netloc', and 'path'.

    Raises:
        ValueError: If the URL is invalid.
    """
    if not isinstance(url, str):
        raise ValueError("URL must be a string")

    url = url.strip()
    if not url:
        raise ValueError("URL is empty")

    parsed = urlparse(url)

    if not parsed.scheme:
        raise ValueError("URL is missing a scheme")

    if parsed.scheme.lower() == "file":
        # file URLs may have empty netloc but must have a path
        if not parsed.path:
            raise ValueError("File URL must include a path")
    else:
        if not parsed.netloc:
            raise ValueError("URL is missing a network location (netloc/host)")

    return {
        "scheme": parsed.scheme,
        "netloc": parsed.netloc,
        "path": parsed.path,
    }


def generate_query_string(params: dict) -> str:
    """
    Generate a URL-encoded query string from a dictionary of parameters.

    - Omits keys with value None.
    - Expands list/tuple values into multiple key=value pairs (doseq=True).

    Args:
        params (dict): Dictionary of query parameters.

    Returns:
        str: URL-encoded query string (without leading '?').

    Raises:
        ValueError: If params is not a dictionary or contains nested dictionaries.
    """
    if params is None:
        return ""
    if not isinstance(params, dict):
        raise ValueError("params must be a dictionary")

    filtered = {}
    for key, value in params.items():
        if value is None:
            continue
        if isinstance(value, dict):
            raise ValueError("Nested dictionaries are not supported in query parameters")
        filtered[key] = value

    if not filtered:
        return ""

    return urlencode(filtered, doseq=True)


def create_redirect_url(base_url_api: str, desired_endpoint: str, optional_params: dict = None) -> str:
    """
    Create a validated redirect URL for the API.

    Args:
        base_url_api (str): Base API URL (e.g., "https://api.myservice.com").
        desired_endpoint (str): Endpoint path that must be in ALLOWED_ENDPOINTS.
        optional_params (dict): Optional query parameters to append.

    Returns:
        str: Complete redirect URL.

    Raises:
        ValueError: If the base URL or endpoint are invalid, or params invalid.
    """
    if not isinstance(base_url_api, str) or not base_url_api.strip():
        raise ValueError("base_url_api must be a non-empty string")
    if not isinstance(desired_endpoint, str) or not desired_endpoint.strip():
        raise ValueError("desired_endpoint must be a non-empty string")
    if optional_params is not None and not isinstance(optional_params, dict):
        raise ValueError("optional_params must be a dictionary or None")

    base_url_api = base_url_api.strip()
    desired_endpoint = desired_endpoint.strip()

    parsed = urllib.parse.urlparse(base_url_api)
    if not parsed.scheme or not parsed.netloc:
        raise ValueError("Invalid base_url_api; must include scheme and network location")

    if parsed.query or parsed.fragment:
        raise ValueError("base_url_api must not include query string or fragment")

    hostname = parsed.hostname
    if not hostname:
        raise ValueError("Invalid base_url_api; unable to determine hostname")
    if hostname.lower() not in APPROVED_API_DOMAINS:
        raise ValueError("Base domain is not approved")

    if not desired_endpoint.startswith("/"):
        desired_endpoint = "/" + desired_endpoint

    if desired_endpoint not in ALLOWED_ENDPOINTS:
        raise ValueError("Endpoint is not allowed")

    origin = f"{parsed.scheme}://{parsed.netloc}"
    url = urllib.parse.urljoin(origin, desired_endpoint)

    query = generate_query_string(optional_params) if optional_params else ""
    if query:
        url = f"{url}?{query}"

    return url
