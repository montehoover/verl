import urllib.parse

APPROVED_API_DOMAINS = {'api.myservice.com', 'api-test.myservice.com', 'api-staging.myservice.com'}
ALLOWED_ENDPOINTS = ['/v1/data', '/v1/user', '/v2/analytics', '/health']

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
    parsed_base_url = urllib.parse.urlparse(base_url)
    if parsed_base_url.hostname not in APPROVED_API_DOMAINS:
        raise ValueError(f"Domain '{parsed_base_url.hostname}' is not an approved API domain.")

    if path not in ALLOWED_ENDPOINTS:
        raise ValueError(f"Endpoint '{path}' is not an allowed API endpoint.")

    # Ensure base_url ends with a slash for proper joining, if it doesn't have a path component
    # If base_url already has a path (e.g. https://api.myservice.com/api), urljoin handles it.
    # If base_url is just (e.g. https://api.myservice.com), we want it to become https://api.myservice.com/
    # so that urljoin(base, '/v1/data') becomes https://api.myservice.com/v1/data
    # and not https://api.myservice.comv1/data
    effective_base_url = base_url
    if not parsed_base_url.path and not base_url.endswith('/'):
        effective_base_url += '/'
    
    # Join base_url and path. urllib.parse.urljoin handles leading/trailing slashes correctly.
    # e.g. urljoin('http://example.com/foo/', '/bar') -> 'http://example.com/bar'
    # e.g. urljoin('http://example.com/foo', 'bar') -> 'http://example.com/bar'
    # e.g. urljoin('http://example.com/foo/', 'bar') -> 'http://example.com/foo/bar'
    # e.g. urljoin('http://example.com/foo', '/bar') -> 'http://example.com/bar'
    # For our case, path always starts with '/', so it will replace any existing path in base_url
    # which is what we want if base_url was like 'https://api.myservice.com/someprefix'
    # and path is '/v1/data'. The result should be 'https://api.myservice.com/v1/data'.
    # However, if base_url is 'https://api.myservice.com/', urljoin will correctly append.
    full_url_without_params = urllib.parse.urljoin(effective_base_url, path)

    if params:
        query_string = urllib.parse.urlencode(params)
        return f"{full_url_without_params}?{query_string}"
    else:
        return full_url_without_params
