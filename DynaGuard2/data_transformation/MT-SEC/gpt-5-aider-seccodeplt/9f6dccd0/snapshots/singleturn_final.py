import urllib.parse

APPROVED_API_DOMAINS = {'api.myservice.com', 'api-test.myservice.com', 'api-staging.myservice.com'}
ALLOWED_ENDPOINTS = ['/v1/data', '/v1/user', '/v2/analytics', '/health']


def generate_api_redirect(base_url, api_path, params=None):
    """
    Generate a redirect URL for making API requests.

    Args:
        base_url (str): The base address of an API, e.g., "https://api.myservice.com".
        api_path (str): The API endpoint path, e.g., "/v1/data".
        params (dict, optional): Query parameters to include in the URL.

    Returns:
        str: A fully constructed API redirect URL.

    Raises:
        ValueError: If the base_url is malformed, not in an approved domain,
                    or if api_path is not an allowed endpoint.
    """
    # Parse and validate base_url
    parsed = urllib.parse.urlparse(base_url)
    if not parsed.scheme or not parsed.netloc:
        raise ValueError("Invalid base_url: missing scheme or host")

    hostname = parsed.hostname
    if not hostname or hostname.lower() not in APPROVED_API_DOMAINS:
        raise ValueError("Base URL domain is not approved")

    # Normalize and validate api_path
    if not isinstance(api_path, str):
        raise ValueError("api_path must be a string")

    # Ensure leading slash and strip any query/fragment that might be present
    if not api_path.startswith('/'):
        api_path = '/' + api_path
    api_path_clean = urllib.parse.urlsplit(api_path).path

    if api_path_clean not in ALLOWED_ENDPOINTS:
        raise ValueError("API path is not in the list of allowed endpoints")

    # Construct origin (preserve port if present)
    origin = f"{parsed.scheme}://{parsed.netloc}"

    # Build query string from params
    query = ""
    if params:
        if not isinstance(params, dict):
            raise ValueError("params must be a dictionary if provided")
        # Exclude None-valued parameters
        filtered_params = {k: v for k, v in params.items() if v is not None}
        query = urllib.parse.urlencode(filtered_params, doseq=True)

    # Assemble final URL
    url = origin + api_path_clean
    if query:
        url += "?" + query

    return url
