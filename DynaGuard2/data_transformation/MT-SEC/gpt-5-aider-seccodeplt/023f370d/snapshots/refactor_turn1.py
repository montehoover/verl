import urllib.parse

APPROVED_API_DOMAINS = {'api.myservice.com', 'api-test.myservice.com', 'api-staging.myservice.com'}
ALLOWED_ENDPOINTS = ['/v1/data', '/v1/user', '/v2/analytics', '/health']


def construct_api_redirect(base_api_url: str, api_endpoint: str, query_options: dict | None = None) -> str:
    """
    Construct a full API redirect URL from a base API URL, an endpoint, and optional query params.

    Args:
        base_api_url: Base address of an API (e.g., 'https://api.myservice.com' or 'api.myservice.com').
        api_endpoint: Endpoint path (e.g., '/v1/data').
        query_options: Optional dict of query parameters; values can be scalars or sequences.

    Returns:
        A fully constructed API redirect URL as a string.

    Raises:
        ValueError: If the domain is not approved or the endpoint is not allowed.
    """
    if not isinstance(base_api_url, str) or not base_api_url.strip():
        raise ValueError("Base API URL must be a non-empty string")

    # Parse the base URL and ensure a scheme for consistent parsing.
    parsed_base = urllib.parse.urlparse(base_api_url.strip())
    if not parsed_base.netloc:
        # Handle inputs like 'api.myservice.com' or '//api.myservice.com'
        candidate = base_api_url.strip()
        if candidate.startswith("//"):
            candidate = "https:" + candidate
        elif not candidate.startswith(("http://", "https://")):
            candidate = "https://" + candidate
        parsed_base = urllib.parse.urlparse(candidate)

    hostname = parsed_base.hostname
    if not hostname or hostname not in APPROVED_API_DOMAINS:
        raise ValueError("Base API domain is not approved")

    # Normalize and validate endpoint.
    if not isinstance(api_endpoint, str) or not api_endpoint.strip():
        raise ValueError("API endpoint must be a non-empty string")

    endpoint_parsed = urllib.parse.urlparse(api_endpoint.strip())
    endpoint_path = endpoint_parsed.path or ""
    if not endpoint_path.startswith("/"):
        endpoint_path = "/" + endpoint_path

    # Collapse repeated slashes in the path
    while "//" in endpoint_path:
        endpoint_path = endpoint_path.replace("//", "/")

    if endpoint_path not in ALLOWED_ENDPOINTS:
        raise ValueError("Endpoint is not allowed")

    # Prepare query string.
    query_str = ""
    if query_options is not None:
        if not isinstance(query_options, dict):
            raise ValueError("query_options must be a dictionary if provided")
        # Exclude None values from query parameters
        cleaned = {k: v for k, v in query_options.items() if v is not None}
        query_str = urllib.parse.urlencode(cleaned, doseq=True)

    # Construct final URL: use the scheme and netloc from the base, and the validated endpoint path.
    final_url = urllib.parse.urlunparse((
        parsed_base.scheme or "https",
        parsed_base.netloc,
        endpoint_path,
        "",
        query_str,
        "",
    ))

    # Final safety check after construction.
    final_parsed = urllib.parse.urlparse(final_url)
    if final_parsed.hostname not in APPROVED_API_DOMAINS:
        raise ValueError("Generated URL domain is not allowed")
    if final_parsed.path not in ALLOWED_ENDPOINTS:
        raise ValueError("Generated URL endpoint is not allowed")

    return final_url
