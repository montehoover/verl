import urllib.parse
from urllib.parse import urlparse

APPROVED_DOMAINS = {'example.com', 'test.com', 'myservice.com'}

# API-specific approvals
APPROVED_API_DOMAINS = {'api.myservice.com', 'api-test.myservice.com', 'api-staging.myservice.com'}
ALLOWED_ENDPOINTS = ['/v1/data', '/v1/user', '/v2/analytics', '/health']


def validate_url_domain(url: str) -> bool:
    """
    Legacy helper: Returns True if the URL's hostname matches exactly one of the approved domains, else False.
    """
    if not isinstance(url, str) or not url:
        return False

    parsed = urlparse(url)
    if not parsed.netloc:
        parsed = urlparse(f"http://{url}")

    hostname = parsed.hostname
    if not hostname:
        return False

    hostname = hostname.lower().strip(".")
    return hostname in APPROVED_DOMAINS


def validate_url_and_endpoint(url: str, endpoint: str) -> bool:
    """
    Returns True only if:
      - The URL's hostname matches exactly one of the APPROVED_DOMAINS.
      - The provided endpoint is in ALLOWED_ENDPOINTS.
      - The URL's path (ignoring any query parameters) exactly matches the provided endpoint.
    """
    if not isinstance(url, str) or not url:
        return False
    if not isinstance(endpoint, str) or not endpoint:
        return False

    parsed = urlparse(url)
    if not parsed.netloc:
        parsed = urlparse(f"http://{url}")

    hostname = parsed.hostname
    if not hostname:
        return False

    hostname = hostname.lower().strip(".")
    if hostname not in APPROVED_DOMAINS:
        return False

    # Ignore query parameters from URL by using parsed.path only
    url_path = parsed.path or ""

    if endpoint not in ALLOWED_ENDPOINTS:
        return False

    return url_path == endpoint


def construct_redirect_url(main_url: str, endpoint_name: str, params_dict=None) -> str:
    """
    Construct a complete API redirect URL.

    Args:
        main_url: Base API URL (e.g., "https://api.myservice.com").
        endpoint_name: API endpoint (must be in ALLOWED_ENDPOINTS).
        params_dict: Optional dict of query parameters.

    Returns:
        The complete redirect URL as a string.

    Raises:
        ValueError: If the base URL domain is not approved or the endpoint is not allowed,
                    or if inputs are invalid.
    """
    if not isinstance(main_url, str) or not main_url.strip():
        raise ValueError("main_url must be a non-empty string.")
    if not isinstance(endpoint_name, str) or not endpoint_name.strip():
        raise ValueError("endpoint_name must be a non-empty string.")
    if params_dict is not None and not isinstance(params_dict, dict):
        raise ValueError("params_dict must be a dict or None.")

    parsed = urllib.parse.urlparse(main_url)
    if not parsed.netloc:
        parsed = urllib.parse.urlparse(f"https://{main_url}")

    hostname = parsed.hostname
    if not hostname:
        raise ValueError("Invalid base URL.")
    hostname = hostname.lower().strip(".")
    if hostname not in APPROVED_API_DOMAINS:
        raise ValueError("Base URL domain is not in the approved API domains.")

    endpoint = endpoint_name if endpoint_name.startswith("/") else f"/{endpoint_name}"
    if endpoint not in ALLOWED_ENDPOINTS:
        raise ValueError("Endpoint is not allowed.")

    query = ""
    if params_dict:
        query = urllib.parse.urlencode(params_dict, doseq=True)

    new_parts = parsed._replace(path=endpoint, params="", query=query, fragment="")
    return urllib.parse.urlunparse(new_parts)
