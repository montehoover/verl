import urllib.parse
from urllib.parse import urlencode, urlparse

APPROVED_API_DOMAINS = {'api.myservice.com', 'api-test.myservice.com', 'api-staging.myservice.com'}
ALLOWED_ENDPOINTS = ['/v1/data', '/v1/user', '/v2/analytics', '/health']

def build_url_with_query(base_url: str, path: str, query_params: dict = None) -> str:
    """
    Constructs a URL from a base URL, a path, and optional query parameters.

    Args:
        base_url: The base URL (e.g., "http://example.com").
                  Must be a valid URL.
        path: The path component (e.g., "/users").
        query_params: A dictionary of query parameters (e.g., {"id": "123", "type": "user"}).
                      Defaults to None.

    Returns:
        The full URL as a string.

    Raises:
        ValueError: If the base_url is not a valid URL.
    """
    parsed_base = urlparse(base_url)
    if not all([parsed_base.scheme, parsed_base.netloc]):
        raise ValueError(f"Invalid base_url: {base_url}")

    # Ensure the path starts with a slash if it's not empty and base_url doesn't end with one
    if path and not path.startswith('/') and not base_url.endswith('/'):
        full_path = '/' + path
    elif path.startswith('/') and base_url.endswith('/'):
        full_path = path[1:]
    else:
        full_path = path

    url = base_url.rstrip('/') + full_path

    if query_params:
        query_string = urlencode(query_params)
        url += '?' + query_string
    return url

def construct_api_redirect(base_api_url: str, api_endpoint: str, query_options: dict = None) -> str:
    """
    Constructs an API redirect URL with validation against approved domains and endpoints.

    Args:
        base_api_url: The base API URL (e.g., "https://api.myservice.com").
                      Must be a valid URL and its domain must be in APPROVED_API_DOMAINS.
        api_endpoint: The specific API endpoint (e.g., "/v1/data").
                      Must be in ALLOWED_ENDPOINTS.
        query_options: A dictionary of optional query parameters.

    Returns:
        The full API redirect URL as a string.

    Raises:
        ValueError: If base_api_url is invalid, its domain is not approved,
                    or api_endpoint is not allowed.
    """
    parsed_url = urllib.parse.urlparse(base_api_url)

    if not all([parsed_url.scheme, parsed_url.netloc]):
        raise ValueError(f"Invalid base_api_url: {base_api_url}. Must include scheme and netloc.")

    if parsed_url.netloc not in APPROVED_API_DOMAINS:
        raise ValueError(f"Domain '{parsed_url.netloc}' is not an approved API domain.")

    if api_endpoint not in ALLOWED_ENDPOINTS:
        raise ValueError(f"Endpoint '{api_endpoint}' is not an allowed API endpoint.")

    # Ensure the api_endpoint starts with a slash if it's not empty
    # and base_api_url doesn't end with one.
    # Also handle cases where both or neither have slashes at the join point.
    if base_api_url.endswith('/') and api_endpoint.startswith('/'):
        path_segment = api_endpoint[1:]
    elif not base_api_url.endswith('/') and not api_endpoint.startswith('/'):
        path_segment = '/' + api_endpoint if api_endpoint else ''
    else:
        path_segment = api_endpoint

    # Use urljoin to correctly combine base URL and path, handling slashes.
    # We need to ensure base_api_url ends with a '/' for urljoin to work as expected with paths.
    base_for_join = base_api_url if base_api_url.endswith('/') else base_api_url + '/'
    # And ensure api_endpoint does not start with '/' if base_for_join ends with '/'
    endpoint_for_join = api_endpoint[1:] if api_endpoint.startswith('/') else api_endpoint

    full_url_parts = urllib.parse.urlsplit(base_for_join)
    # Reconstruct the base part without any existing path or query
    # This ensures we are only using the scheme and netloc from base_api_url
    # and then appending the validated api_endpoint.
    # However, urljoin is generally better for this.
    # Let's refine the joining logic.

    # The previous path joining logic was a bit complex. urllib.parse.urljoin is more robust.
    # Ensure base_api_url ends with a slash for urljoin to correctly append the path.
    if not base_api_url.endswith('/'):
        base_api_url_for_join = base_api_url + '/'
    else:
        base_api_url_for_join = base_api_url
    
    # Ensure api_endpoint does not start with a slash if base_api_url_for_join already has one.
    # urljoin handles this correctly if the path part is relative.
    final_path = api_endpoint.lstrip('/')

    # Construct the URL without query parameters first
    url_without_query = urllib.parse.urljoin(base_api_url_for_join, final_path)

    if query_options:
        query_string = urllib.parse.urlencode(query_options)
        # Use urlparse, urlunparse to add query parameters correctly
        url_parts = list(urllib.parse.urlsplit(url_without_query))
        url_parts[3] = query_string
        return urllib.parse.urlunsplit(url_parts)
    else:
        return url_without_query
