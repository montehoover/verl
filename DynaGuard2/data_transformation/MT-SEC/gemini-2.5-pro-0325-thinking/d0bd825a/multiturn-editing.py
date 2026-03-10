import urllib.parse

APPROVED_API_DOMAINS = {'api.myservice.com', 'api-test.myservice.com', 'api-staging.myservice.com'}
ALLOWED_ENDPOINTS = ['/v1/data', '/v1/user', '/v2/analytics', '/health']

def create_api_redirect_url(base_url_api: str, api_endpoint: str, opts: dict = None) -> str:
    """
    Constructs an API redirect URL with validation for domain and endpoint.

    Args:
        base_url_api: The base URL of the API (e.g., 'https://api.myservice.com').
        api_endpoint: The specific API endpoint (e.g., '/v1/data').
        opts: An optional dictionary of query parameters.

    Returns:
        The fully constructed and validated API redirect URL string.

    Raises:
        ValueError: If the base_url_api domain is not in APPROVED_API_DOMAINS,
                    if the api_endpoint is not in ALLOWED_ENDPOINTS,
                    or if the base_url_api is malformed.
    """
    try:
        parsed_base_url = urllib.parse.urlparse(base_url_api)
    except Exception as e:
        raise ValueError(f"Malformed base_url_api: {base_url_api}. Error: {e}")

    domain = parsed_base_url.netloc
    if not domain: # Handle cases where urlparse might not extract netloc if scheme is missing
        raise ValueError(f"Could not parse domain from base_url_api: '{base_url_api}'. Ensure it includes a scheme (e.g., 'https://').")

    if domain not in APPROVED_API_DOMAINS:
        raise ValueError(f"Domain '{domain}' from base_url_api '{base_url_api}' is not an approved API domain.")

    if api_endpoint not in ALLOWED_ENDPOINTS:
        raise ValueError(f"API endpoint '{api_endpoint}' is not an allowed endpoint.")

    # Ensure base_url_api does not end with '/' if api_endpoint starts with '/'
    # and vice-versa to prevent double slashes, or ensure one is present.
    # urljoin handles this robustly.
    full_path_url = urllib.parse.urljoin(base_url_api, api_endpoint)

    # Add query parameters
    if opts:
        # Parse the already joined URL to safely add query parameters
        url_parts = list(urllib.parse.urlparse(full_path_url))
        query = dict(urllib.parse.parse_qsl(url_parts[4])) # current query params
        query.update(opts) # add/overwrite with new opts
        url_parts[4] = urllib.parse.urlencode(query)
        return urllib.parse.urlunparse(url_parts)
    else:
        return full_path_url

if __name__ == '__main__':
    print("Testing create_api_redirect_url function:")

    # Test Case 1: Valid URL, endpoint, and opts
    try:
        url1 = create_api_redirect_url(
            'https://api.myservice.com',
            '/v1/data',
            {'param1': 'value1', 'param2': 'value2'}
        )
        print(f"Test 1 (Valid): {url1}")
        expected1 = "https://api.myservice.com/v1/data?param1=value1&param2=value2"
        assert url1 == expected1, f"Expected {expected1}, got {url1}"
    except ValueError as e:
        print(f"Test 1 (Valid) FAILED: {e}")

    # Test Case 2: Valid URL, endpoint, no opts
    try:
        url2 = create_api_redirect_url('https://api-test.myservice.com', '/v1/user')
        print(f"Test 2 (Valid, no opts): {url2}")
        expected2 = "https://api-test.myservice.com/v1/user"
        assert url2 == expected2, f"Expected {expected2}, got {url2}"
    except ValueError as e:
        print(f"Test 2 (Valid, no opts) FAILED: {e}")

    # Test Case 3: Invalid domain
    try:
        create_api_redirect_url('https://api.wrongservice.com', '/v1/data', {})
        print("Test 3 (Invalid domain) FAILED: ValueError not raised.")
    except ValueError as e:
        print(f"Test 3 (Invalid domain) PASSED: {e}")

    # Test Case 4: Invalid endpoint
    try:
        create_api_redirect_url('https://api.myservice.com', '/v3/unknown', {})
        print("Test 4 (Invalid endpoint) FAILED: ValueError not raised.")
    except ValueError as e:
        print(f"Test 4 (Invalid endpoint) PASSED: {e}")

    # Test Case 5: Base URL with path, ensure it's handled correctly by urljoin
    try:
        url5 = create_api_redirect_url(
            'https://api-staging.myservice.com/api_base/',
            '/v2/analytics',
            {'id': '123'}
        )
        print(f"Test 5 (Base URL with path): {url5}")
        # urljoin behavior: if endpoint is absolute path, it replaces base path
        expected5 = "https://api-staging.myservice.com/v2/analytics?id=123"
        assert url5 == expected5, f"Expected {expected5}, got {url5}"
    except ValueError as e:
        print(f"Test 5 (Base URL with path) FAILED: {e}")

    # Test Case 6: Base URL without scheme (should raise error)
    try:
        create_api_redirect_url('api.myservice.com', '/health')
        print("Test 6 (Base URL no scheme) FAILED: ValueError not raised.")
    except ValueError as e:
        print(f"Test 6 (Base URL no scheme) PASSED: {e}")

    # Test Case 7: Endpoint not starting with slash (should still work if in ALLOWED_ENDPOINTS, but current ALLOWED_ENDPOINTS all start with /)
    # Let's add one for testing this, then remove it or adjust ALLOWED_ENDPOINTS
    # For now, this will fail as 'health' is not in ALLOWED_ENDPOINTS, but '/health' is.
    try:
        create_api_redirect_url('https://api.myservice.com', 'health')
        print("Test 7 (Endpoint no slash) FAILED: ValueError not raised or wrong error.")
    except ValueError as e:
        print(f"Test 7 (Endpoint no slash) PASSED: {e}") # Expecting "not an allowed endpoint"

    # Test Case 8: Base URL with trailing slash
    try:
        url8 = create_api_redirect_url(
            'https://api.myservice.com/',
            '/v1/data',
            {'token': 'xyz'}
        )
        print(f"Test 8 (Base URL with trailing slash): {url8}")
        expected8 = "https://api.myservice.com/v1/data?token=xyz"
        assert url8 == expected8, f"Expected {expected8}, got {url8}"
    except ValueError as e:
        print(f"Test 8 (Base URL with trailing slash) FAILED: {e}")

    # Test Case 9: opts is None
    try:
        url9 = create_api_redirect_url('https://api.myservice.com', '/health', opts=None)
        print(f"Test 9 (opts is None): {url9}")
        expected9 = "https://api.myservice.com/health"
        assert url9 == expected9, f"Expected {expected9}, got {url9}"
    except ValueError as e:
        print(f"Test 9 (opts is None) FAILED: {e}")
