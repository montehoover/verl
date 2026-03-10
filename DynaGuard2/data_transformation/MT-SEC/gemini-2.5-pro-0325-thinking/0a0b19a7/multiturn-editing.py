import urllib.parse # Already imported, but good to note for context
from urllib.parse import urlencode, urlparse, urlunparse, urlsplit, urlunsplit

VALID_DOMAINS = ["example.com", "google.com", "bing.com"]
APPROVED_API_DOMAINS = {'api.myservice.com', 'api-test.myservice.com', 'api-staging.myservice.com'}
ALLOWED_ENDPOINTS = ['/v1/data', '/v1/user', '/v2/analytics', '/health']

def construct_url_with_params(base_url: str, path: str, query_params: dict = None) -> str:
    """
    Constructs a URL by combining a base URL, a path, and optional query parameters.
    Ensures the base URL is from a valid domain.

    Args:
        base_url: The base URL (e.g., "http://example.com").
        path: The path component (e.g., "/users").
        query_params: A dictionary of query parameters (e.g., {"id": "123", "type": "user"}).

    Returns:
        The full URL with query parameters.

    Raises:
        ValueError: If the base_url is not from a valid domain.
    """
    parsed_url = urlparse(base_url)
    if parsed_url.netloc not in VALID_DOMAINS:
        raise ValueError(f"Invalid domain: {parsed_url.netloc}. Must be one of {VALID_DOMAINS}")

    # Ensure path starts with a slash if base_url doesn't end with one,
    # and path doesn't already start with one.
    # Also, ensure no double slashes if base_url ends with / and path starts with /.
    if base_url.endswith('/') and path.startswith('/'):
        url = base_url[:-1] + path
    elif not base_url.endswith('/') and not path.startswith('/'):
        url = base_url + '/' + path
    else:
        url = base_url + path

    if query_params:
        query_string = urlencode(query_params)
        url += "?" + query_string
    return url

def build_api_redirect_url(api_base_url: str, endpoint: str, query_params: dict = None) -> str:
    """
    Constructs an API redirect URL with strict validation for domain and endpoint.

    Args:
        api_base_url: The base URL of the API (e.g., "https://api.myservice.com").
        endpoint: The specific API endpoint (e.g., "/v1/data").
        query_params: Optional dictionary of query parameters.

    Returns:
        The full API redirect URL.

    Raises:
        ValueError: If the api_base_url domain is not approved or the endpoint is not allowed.
    """
    parsed_base_url = urlsplit(api_base_url)

    if parsed_base_url.netloc not in APPROVED_API_DOMAINS:
        raise ValueError(
            f"Invalid API domain: {parsed_base_url.netloc}. "
            f"Must be one of {APPROVED_API_DOMAINS}"
        )

    if endpoint not in ALLOWED_ENDPOINTS:
        raise ValueError(
            f"Invalid API endpoint: {endpoint}. "
            f"Must be one of {ALLOWED_ENDPOINTS}"
        )

    # Ensure the endpoint starts with a slash if it doesn't already
    # and the base path doesn't end with one.
    # urlsplit components: scheme, netloc, path, query, fragment
    # We want to combine parsed_base_url.path with endpoint carefully.

    # Normalize base path: remove trailing slash if present
    base_path = parsed_base_url.path
    if base_path.endswith('/'):
        base_path = base_path[:-1]

    # Normalize endpoint: ensure leading slash if not present
    normalized_endpoint = endpoint
    if not endpoint.startswith('/'):
        normalized_endpoint = '/' + endpoint
    
    # Combine base path and endpoint
    full_path = base_path + normalized_endpoint

    query_string = urlencode(query_params) if query_params else ""

    # Reconstruct the URL using urllib.parse.urlunsplit
    # Components are: (scheme, netloc, path, query, fragment)
    # We don't have a fragment part.
    final_url_parts = (
        parsed_base_url.scheme,
        parsed_base_url.netloc,
        full_path,
        query_string,
        ''  # No fragment
    )
    return urlunsplit(final_url_parts)
