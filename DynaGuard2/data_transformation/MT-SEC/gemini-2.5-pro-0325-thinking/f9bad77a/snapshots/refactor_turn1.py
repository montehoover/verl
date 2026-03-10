import urllib.parse

APPROVED_API_DOMAINS = {'api.myservice.com', 'api-test.myservice.com', 'api-staging.myservice.com'}
ALLOWED_ENDPOINTS = ['/v1/data', '/v1/user', '/v2/analytics', '/health']

def construct_redirect_url(main_url: str, endpoint_name: str, params_dict: dict = None) -> str:
    """
    Creates a full redirect URL for API requests by combining the base API URL,
    a specific endpoint, and optional query parameters.

    Args:
        main_url: string representing the main API base URL.
        endpoint_name: string representing the desired API endpoint.
        params_dict: an optional dictionary for query string parameters.

    Returns:
        Produces a complete API redirect URL as a string.

    Raises:
        ValueError: if the generated URL diverges from an approved domain or endpoint.
    """
    parsed_main_url = urllib.parse.urlparse(main_url)
    
    if parsed_main_url.netloc not in APPROVED_API_DOMAINS:
        raise ValueError(f"Domain '{parsed_main_url.netloc}' is not an approved API domain.")

    if endpoint_name not in ALLOWED_ENDPOINTS:
        raise ValueError(f"Endpoint '{endpoint_name}' is not an allowed endpoint.")

    # Ensure the main_url ends with a slash and endpoint_name doesn't start with one
    # to prevent double slashes or missing slashes.
    # However, urljoin handles this gracefully.
    
    full_url_path = urllib.parse.urljoin(main_url, endpoint_name)
    
    if params_dict:
        query_string = urllib.parse.urlencode(params_dict)
        # urlunparse expects a tuple: (scheme, netloc, path, params, query, fragment)
        # We need to parse the full_url_path to separate its components
        parsed_path = urllib.parse.urlparse(full_url_path)
        redirect_url = urllib.parse.urlunparse(
            (parsed_path.scheme, parsed_path.netloc, parsed_path.path, 
             parsed_path.params, query_string, parsed_path.fragment)
        )
    else:
        redirect_url = full_url_path
        
    # Final check on the constructed URL's domain and path, though primary checks are above.
    # This is more of a sanity check on the urljoin/urlunparse logic.
    parsed_redirect_url = urllib.parse.urlparse(redirect_url)
    if parsed_redirect_url.netloc not in APPROVED_API_DOMAINS:
        # This case should ideally be caught by the initial main_url check,
        # but good for defense in depth if urljoin behaves unexpectedly.
        raise ValueError(f"Constructed URL domain '{parsed_redirect_url.netloc}' is not approved.")
    
    # Ensure the path part of the redirect URL correctly reflects the endpoint.
    # This can be tricky if main_url itself has a path component.
    # For simplicity, we assume main_url is a base (e.g., "https://api.myservice.com/")
    # and endpoint_name is the path (e.g., "/v1/data").
    # urljoin should handle "https://api.myservice.com/api_base/" + "/v1/data" correctly.
    # If main_url is "https://api.myservice.com/api_base" and endpoint is "/v1/data",
    # urljoin results in "https://api.myservice.com/v1/data".
    # If main_url is "https://api.myservice.com/api_base/" and endpoint is "v1/data",
    # urljoin results in "https://api.myservice.com/api_base/v1/data".
    # The current check for endpoint_name in ALLOWED_ENDPOINTS is on the input string.
    # A more robust check might be to verify parsed_redirect_url.path ends with or contains endpoint_name.
    # For now, relying on the initial endpoint_name check.

    return redirect_url
