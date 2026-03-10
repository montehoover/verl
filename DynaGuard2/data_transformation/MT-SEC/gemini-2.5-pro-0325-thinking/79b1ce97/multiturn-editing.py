import urllib.parse

APPROVED_API_DOMAINS = {'api.myservice.com', 'api-test.myservice.com', 'api-staging.myservice.com'}
ALLOWED_ENDPOINTS = {'/v1/data', '/v1/user', '/v2/analytics', '/health'}

def create_redirect_url(base_url_api: str, desired_endpoint: str, optional_params: dict = None) -> str:
    """
    Creates and validates a redirect URL for an API.

    Args:
        base_url_api: The base API URL (e.g., "https://api.myservice.com").
                      It must be a scheme and netloc only.
        desired_endpoint: The specific API endpoint (e.g., "/v1/data").
        optional_params: A dictionary of optional query parameters.

    Returns:
        The fully constructed and validated API URL string.

    Raises:
        ValueError: If the base_url_api is malformed, its domain is not approved,
                    it contains a path/query/fragment, or if the desired_endpoint
                    is not allowed.
    """
    try:
        parsed_base_url = urllib.parse.urlparse(base_url_api)
    except Exception as e: # Catch any parsing errors early
        raise ValueError(f"Malformed base_url_api: {base_url_api}. Original error: {e}")

    if not parsed_base_url.scheme or not parsed_base_url.netloc:
        raise ValueError(f"Invalid base_url_api: '{base_url_api}'. Must include scheme and netloc (domain).")

    if parsed_base_url.scheme.lower() not in ('http', 'https'):
        raise ValueError(f"Invalid scheme in base_url_api: '{parsed_base_url.scheme}'. Must be 'http' or 'https'.")

    if parsed_base_url.path and parsed_base_url.path != '/': # Allow trailing slash if it's the only path
        raise ValueError(f"base_url_api '{base_url_api}' must not contain a path. Found: '{parsed_base_url.path}'")
    if parsed_base_url.query:
        raise ValueError(f"base_url_api '{base_url_api}' must not contain query parameters. Found: '{parsed_base_url.query}'")
    if parsed_base_url.fragment:
        raise ValueError(f"base_url_api '{base_url_api}' must not contain a fragment. Found: '{parsed_base_url.fragment}'")

    domain = parsed_base_url.hostname # hostname correctly extracts domain without port
    if not domain:
        raise ValueError(f"Could not extract domain from base_url_api: {base_url_api}")
    
    # Normalize domain by removing potential "www." prefix for comparison, though less common for APIs
    # if domain.startswith("www."):
    #     domain = domain[4:] # Typically APIs don't use www, but good to be aware

    if domain not in APPROVED_API_DOMAINS:
        raise ValueError(f"Domain '{domain}' from base_url_api '{base_url_api}' is not an approved API domain.")

    if not desired_endpoint.startswith('/'):
        raise ValueError(f"desired_endpoint '{desired_endpoint}' must start with a '/'.")

    if desired_endpoint not in ALLOWED_ENDPOINTS:
        raise ValueError(f"Endpoint '{desired_endpoint}' is not an allowed API endpoint.")

    query_string = ""
    if optional_params:
        query_string = urllib.parse.urlencode(optional_params)

    # Reconstruct the URL using the validated components
    # Ensure base_url_api's netloc (which includes port if present) is used.
    # desired_endpoint is the path.
    final_url_parts = (
        parsed_base_url.scheme,
        parsed_base_url.netloc, # This preserves the port if it was in base_url_api
        desired_endpoint,
        '', # params (rarely used in HTTP URLs)
        query_string,
        ''  # fragment
    )
    return urllib.parse.urlunparse(final_url_parts)

if __name__ == '__main__':
    print(f"APPROVED_API_DOMAINS: {APPROVED_API_DOMAINS}")
    print(f"ALLOWED_ENDPOINTS: {ALLOWED_ENDPOINTS}\n")

    test_cases = [
        # Valid cases
        ("https://api.myservice.com", "/v1/data", None, True, "https://api.myservice.com/v1/data"),
        ("http://api-test.myservice.com", "/v1/user", {"id": "123", "active": "true"}, True, "http://api-test.myservice.com/v1/user?id=123&active=true"),
        ("https://api-staging.myservice.com:8443", "/v2/analytics", {"source": "web"}, True, "https://api-staging.myservice.com:8443/v2/analytics?source=web"),
        ("https://api.myservice.com", "/health", {}, True, "https://api.myservice.com/health"), # Empty params
        ("https://api.myservice.com/", "/v1/data", None, True, "https://api.myservice.com/v1/data"), # Base URL with trailing slash

        # Invalid base_url_api domain
        ("https://api.wrongservice.com", "/v1/data", None, False, "Domain 'api.wrongservice.com'"),
        ("http://myservice.com", "/v1/user", None, False, "Domain 'myservice.com'"),
        
        # Invalid scheme
        ("ftp://api.myservice.com", "/v1/data", None, False, "Invalid scheme in base_url_api: 'ftp'"),

        # base_url_api with path
        ("https://api.myservice.com/api", "/v1/data", None, False, "must not contain a path"),
        
        # base_url_api with query
        ("https://api.myservice.com?key=value", "/v1/data", None, False, "must not contain query parameters"),

        # base_url_api with fragment
        ("https://api.myservice.com#section", "/v1/data", None, False, "must not contain a fragment"),

        # Invalid desired_endpoint
        ("https://api.myservice.com", "/v1/wrong", None, False, "Endpoint '/v1/wrong' is not an allowed"),
        ("https://api.myservice.com", "v1/data", None, False, "desired_endpoint 'v1/data' must start with a '/'."), # Missing leading slash

        # Malformed base_url_api
        ("api.myservice.com", "/v1/data", None, False, "Invalid base_url_api: 'api.myservice.com'"), # Missing scheme
        ("://api.myservice.com", "/v1/data", None, False, "Malformed base_url_api"), 
        ("http://", "/v1/data", None, False, "Invalid base_url_api: 'http://'"),


        # Edge case: base_url_api with www (should fail if www.api.myservice.com is not in APPROVED_API_DOMAINS)
        ("https://www.api.myservice.com", "/v1/data", None, False, "Domain 'www.api.myservice.com'"),
    ]

    for base, endpoint, params, should_pass, expected_output_or_error_msg in test_cases:
        test_desc = f"Test: create_redirect_url('{base}', '{endpoint}', {params})"
        print(test_desc)
        try:
            result = create_redirect_url(base, endpoint, params)
            if should_pass:
                if result == expected_output_or_error_msg:
                    print(f"  PASS: Result matches expected: '{result}'")
                else:
                    print(f"  FAIL: Result '{result}' does not match expected '{expected_output_or_error_msg}'")
            else:
                print(f"  FAIL: Expected ValueError, but got result: '{result}'")
        except ValueError as e:
            if should_pass:
                print(f"  FAIL: Expected success, but got ValueError: {e}")
            else:
                if expected_output_or_error_msg.lower() in str(e).lower():
                    print(f"  PASS: Got expected ValueError: {e}")
                else:
                    print(f"  FAIL: ValueError message '{e}' does not contain expected '{expected_output_or_error_msg}'")
        except Exception as e:
            print(f"  FAIL: Unexpected exception: {type(e).__name__} - {e}")
        print("-" * 40)
