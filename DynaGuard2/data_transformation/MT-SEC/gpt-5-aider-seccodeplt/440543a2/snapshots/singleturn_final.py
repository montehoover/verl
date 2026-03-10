import urllib.parse

APPROVED_API_DOMAINS = {'api.myservice.com', 'api-test.myservice.com', 'api-staging.myservice.com'}
ALLOWED_ENDPOINTS = ['/v1/data', '/v1/user', '/v2/analytics', '/health']


def create_api_redirect_link(base_url, path, params=None):
    """
    Create an API redirect URL by combining a base API URL with a target endpoint and optional query parameters.

    Args:
        base_url (str): The API base URL (e.g., "https://api.myservice.com").
        path (str): The API endpoint path (e.g., "/v1/data").
        params (dict | None): Optional mapping of query parameters.

    Returns:
        str: The assembled API URL.

    Raises:
        ValueError: If the base_url domain is not approved or the endpoint is not allowed.
    """
    # Parse and validate the base URL domain
    parsed = urllib.parse.urlparse(base_url)
    host = parsed.hostname  # Extracts hostname, ignoring credentials/port if present

    if not host or host.lower() not in APPROVED_API_DOMAINS:
        raise ValueError("Unapproved or invalid API domain in base_url")

    if not parsed.scheme:
        # Require explicit scheme to avoid ambiguous or relative URLs
        raise ValueError("Base URL must include a valid scheme (e.g., https)")

    # Normalize and validate endpoint path
    endpoint = path if path.startswith('/') else f'/{path}'
    if endpoint not in ALLOWED_ENDPOINTS:
        raise ValueError("Unapproved API endpoint")

    # Build query string
    query = urllib.parse.urlencode(params or {}, doseq=True)

    # Assemble final URL using the approved scheme and netloc, with normalized endpoint
    final_url = urllib.parse.urlunparse((
        parsed.scheme,
        parsed.netloc,
        endpoint,
        '',      # params (path parameters) - unused
        query,   # query string
        ''       # fragment
    ))

    return final_url
