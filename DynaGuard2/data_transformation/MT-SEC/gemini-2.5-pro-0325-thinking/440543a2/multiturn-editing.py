from urllib.parse import urlencode
import urllib.parse # Added import

def construct_url_with_params(base_url: str, path: str, query_params: dict = None) -> str:
    """
    Constructs a URL from a base URL, a path, and optional query parameters,
    ensuring the URL uses HTTPS.

    Args:
        base_url: The base URL (e.g., "http://example.com" or "example.com").
        path: The path component (e.g., "/users").
        query_params: A dictionary of query parameters (e.g., {"id": "123", "type": "user"}).

    Returns:
        The full URL with query parameters.
    """
    if not base_url.startswith("https://"):
        if base_url.startswith("http://"):
            base_url = "https://" + base_url[len("http://"):]
        else:
            base_url = "https://" + base_url

    # Ensure the path starts with a slash if it's not empty and base_url doesn't end with one
    if path and not path.startswith("/") and not base_url.endswith("/"):
        full_url = base_url + "/" + path
    # Prevent double slashes if base_url ends with / and path starts with /
    elif base_url.endswith("/") and path.startswith("/"):
        full_url = base_url + path[1:]
    else:
        full_url = base_url + path

    if query_params:
        query_string = urlencode(query_params)
        full_url += "?" + query_string
    
    return full_url

APPROVED_API_DOMAINS = {'api.myservice.com', 'api-test.myservice.com', 'api-staging.myservice.com'}
ALLOWED_ENDPOINTS = ['/v1/data', '/v1/user', '/v2/analytics', '/health']

def create_api_redirect_link(base_url: str, path: str, params: dict = None) -> str:
    """
    Constructs an API redirect URL, validating the base URL and path against approved lists.

    Args:
        base_url: The base address (e.g., "https://api.myservice.com").
        path: The specific API endpoint (e.g., "/v1/data").
        params: A dictionary for optional query parameters.

    Returns:
        The full API redirect URL.

    Raises:
        ValueError: If the base_url domain is not in APPROVED_API_DOMAINS,
                    or if the path is not in ALLOWED_ENDPOINTS,
                    or if the base_url does not start with 'https://'.
    """
    parsed_url = urllib.parse.urlparse(base_url)
    
    if not parsed_url.scheme == "https":
        raise ValueError(f"Invalid base URL: '{base_url}'. Must use 'https'.")

    domain = parsed_url.netloc
    if domain not in APPROVED_API_DOMAINS:
        raise ValueError(f"Invalid domain: '{domain}'. Not in approved API domains.")

    if path not in ALLOWED_ENDPOINTS:
        raise ValueError(f"Invalid path: '{path}'. Not an allowed endpoint.")

    # Ensure base_url ends with a slash if path doesn't start with one, and vice-versa, to avoid double slashes.
    # urllib.parse.urljoin handles this robustly.
    # However, the request is to build it using urllib.parse components.
    # We'll construct the URL parts and then join.

    # Reconstruct base_url to ensure it's just scheme + netloc for urlunparse
    # This also ensures no existing path or query params from base_url are kept if not intended.
    # For this function, base_url is expected to be just the scheme and domain.
    
    # Let's ensure the path is correctly joined.
    # If base_url has a path component, urljoin is better.
    # Given the problem description, base_url is likely "https://domain.com"
    # and path is "/v1/data".

    # We can use urlunparse for a more structured approach if we build all parts.
    # scheme, netloc, path, params, query, fragment
    
    query_string = ""
    if params:
        query_string = urllib.parse.urlencode(params)

    # Construct the final URL using urllib.parse.urlunparse
    # scheme, netloc, path, params (for path segment parameters, typically empty), query, fragment (empty)
    final_url_parts = (parsed_url.scheme, parsed_url.netloc, path, '', query_string, '')
    final_url = urllib.parse.urlunparse(final_url_parts)
    
    return final_url
