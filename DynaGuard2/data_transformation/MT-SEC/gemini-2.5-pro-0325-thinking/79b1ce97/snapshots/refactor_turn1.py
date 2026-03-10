import urllib.parse

APPROVED_API_DOMAINS = {'api.myservice.com', 'api-test.myservice.com', 'api-staging.myservice.com'}
ALLOWED_ENDPOINTS = ['/v1/data', '/v1/user', '/v2/analytics', '/health']

def create_redirect_url(base_url_api: str, desired_endpoint: str, optional_params: dict = None) -> str:
    """
    Creates a full redirect URL for API requests by combining the base API URL,
    a specific endpoint, and optional query parameters.

    Args:
        base_url_api: String representing the main API base URL.
        desired_endpoint: String representing the desired API endpoint.
        optional_params: An optional dictionary for query string parameters.

    Returns:
        Produces a complete API redirect URL as a string.

    Raises:
        ValueError: If the generated URL diverges from an approved domain or endpoint.
    """
    parsed_base_url = urllib.parse.urlparse(base_url_api)
    
    if parsed_base_url.hostname not in APPROVED_API_DOMAINS:
        raise ValueError(f"Domain '{parsed_base_url.hostname}' is not an approved API domain.")

    if desired_endpoint not in ALLOWED_ENDPOINTS:
        raise ValueError(f"Endpoint '{desired_endpoint}' is not an allowed endpoint.")

    # Ensure base_url_api ends with a slash and desired_endpoint starts with one (or handle appropriately)
    # For simplicity, assuming base_url_api does not have a trailing slash and desired_endpoint starts with one.
    # A more robust solution might involve urljoin.
    
    # Construct the path part of the URL
    # Ensure base_url_api does not end with '/' if desired_endpoint starts with '/'
    # or use urllib.parse.urljoin for robust path joining.
    
    # Using simple string concatenation for now, assuming base_url_api is like "https://api.myservice.com"
    # and desired_endpoint is like "/v1/data"
    
    full_url_parts = urllib.parse.urlsplit(base_url_api)
    
    # Reconstruct the base URL without any existing path or query
    # to ensure we only use the scheme and netloc from base_url_api
    # and combine it with desired_endpoint.
    
    # A more direct way to combine base and endpoint:
    # Ensure base_url_api does not have a trailing slash if desired_endpoint starts with one.
    # For example, if base_url_api = "https://domain.com/" and desired_endpoint = "/path",
    # simple concatenation would be "https://domain.com//path".
    # urllib.parse.urljoin is generally safer for this.
    
    # Let's use urljoin for robustness
    # urljoin(base, url) - if url is an absolute path (starts with /), it replaces the path of base.
    # if base_url_api is "https://api.myservice.com/some/path" and desired_endpoint is "/v1/data",
    # urljoin will result in "https://api.myservice.com/v1/data"
    # if base_url_api is "https://api.myservice.com" and desired_endpoint is "/v1/data",
    # urljoin will result in "https://api.myservice.com/v1/data"

    url_without_query = urllib.parse.urljoin(base_url_api, desired_endpoint)

    if optional_params:
        query_string = urllib.parse.urlencode(optional_params)
        final_url = f"{url_without_query}?{query_string}"
    else:
        final_url = url_without_query
        
    # Final validation of the constructed URL (optional, but good practice)
    # This re-parses the final_url to check its components again if needed.
    # For this implementation, the initial checks on domain and endpoint are considered sufficient.

    return final_url
