import urllib.parse
import logging

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

APPROVED_API_DOMAINS = {'api.myservice.com', 'api-test.myservice.com', 'api-staging.myservice.com'}
ALLOWED_ENDPOINTS = ['/v1/data', '/v1/user', '/v2/analytics', '/health']


def _validate_url_components(base_url_str: str, path_str: str, approved_domains: set, allowed_paths: list) -> None:
    """
    Validates the domain from base_url_str and the path_str.

    Args:
        base_url_str: The base URL string.
        path_str: The endpoint path string.
        approved_domains: A set of approved domain names.
        allowed_paths: A list of allowed endpoint paths.

    Raises:
        ValueError: If the domain or path is not approved/allowed.
    """
    parsed_base_url = urllib.parse.urlparse(base_url_str)
    if parsed_base_url.hostname not in approved_domains:
        raise ValueError(f"Domain '{parsed_base_url.hostname}' is not an approved API domain.")

    if path_str not in allowed_paths:
        raise ValueError(f"Endpoint '{path_str}' is not an allowed API endpoint.")


def _construct_url(base_url_str: str, path_str: str, params_dict: dict = None) -> str:
    """
    Constructs the full URL from base, path, and optional parameters.

    Args:
        base_url_str: The base URL string.
        path_str: The endpoint path string.
        params_dict: An optional dictionary of query parameters.

    Returns:
        The fully constructed URL string.
    """
    parsed_base_url = urllib.parse.urlparse(base_url_str) # For checking .path component

    # Ensure base_url_str ends with a slash for proper joining, if it doesn't have a path component.
    # If base_url_str already has a path (e.g., "https://api.myservice.com/api"), urljoin handles it.
    # If base_url_str is just a domain (e.g., "https://api.myservice.com"), we want it to become "https://api.myservice.com/"
    # so that urljoin("https://api.myservice.com/", "/v1/data") results in "https://api.myservice.com/v1/data"
    # and not "https://api.myservice.comv1/data".
    effective_base_url = base_url_str
    if not parsed_base_url.path and not base_url_str.endswith('/'):
        effective_base_url += '/'
    
    # Join effective_base_url and path_str. 
    # urllib.parse.urljoin handles various cases of leading/trailing slashes correctly.
    # For example:
    #   urljoin('http://example.com/foo/', '/bar') -> 'http://example.com/bar' (path replaces)
    #   urljoin('http://example.com/foo/', 'bar')  -> 'http://example.com/foo/bar' (path appends)
    # Since path_str is expected to be an absolute path (e.g., '/v1/data'), it will typically
    # replace any existing path in effective_base_url if effective_base_url itself had a path component
    # (e.g. urljoin("https://api.myservice.com/some/prefix", "/v1/data") -> "https://api.myservice.com/v1/data").
    # If effective_base_url is "https://api.myservice.com/", it correctly appends.
    full_url_without_params = urllib.parse.urljoin(effective_base_url, path_str)

    if params_dict:
        query_string = urllib.parse.urlencode(params_dict)
        return f"{full_url_without_params}?{query_string}"
    else:
        return full_url_without_params


def create_api_redirect_link(base_url: str, path: str, params: dict = None) -> str:
    """
    Creates an API redirect URL by merging a base API URL with a defined endpoint 
    and any optional query parameters.

    Args:
        base_url: A string designating the API’s base URL.
        path: A string identifying the target API endpoint.
        params: An optional dictionary containing key-value pairs used as query parameters.

    Returns:
        A string representing the final, assembled API URL.

    Raises:
        ValueError: Triggered if the constructed URL points to an unapproved domain or endpoint.
    """
    _validate_url_components(base_url, path, APPROVED_API_DOMAINS, ALLOWED_ENDPOINTS)
    constructed_url = _construct_url(base_url, path, params)
    
    logger.info(
        f"URL creation attempt: base_url='{base_url}', path='{path}', params={params}, "
        f"constructed_url='{constructed_url}'"
    )
    
    return constructed_url
