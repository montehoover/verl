from urllib.parse import urlparse, urlencode

APPROVED_DOMAINS = {'example.com', 'test.com', 'myservice.com'} # Remains for validate_url_and_endpoint
APPROVED_API_DOMAINS = {'api.myservice.com', 'api-test.myservice.com', 'api-staging.myservice.com'}
ALLOWED_ENDPOINTS = ['/v1/data', '/v1/user', '/v2/analytics', '/health'] # Updated list, will affect validate_url_and_endpoint and be used by construct_redirect_url

def validate_url_and_endpoint(url: str, endpoint: str) -> bool:
    """
    Validates if the domain of the given URL is in the approved list
    and if the URL's path matches the given endpoint.
    Query parameters are ignored for endpoint validation.

    Args:
        url: The URL string to validate.
        endpoint: The specific endpoint to validate against.

    Returns:
        True if both the domain and endpoint are approved, False otherwise.
    """
    if endpoint not in ALLOWED_ENDPOINTS:
        return False # Early exit if the provided endpoint itself is not allowed

    try:
        parsed_url = urlparse(url)
        domain = parsed_url.netloc
        path = parsed_url.path

        # Handle cases like 'www.example.com' by checking the base domain
        if domain.startswith('www.'):
            domain = domain[4:]

        is_domain_approved = domain in APPROVED_DOMAINS
        # Ensure the path from URL matches the provided endpoint, ignoring query params
        is_endpoint_approved = path == endpoint
        
        return is_domain_approved and is_endpoint_approved
    except Exception:
        # In case of any parsing error, consider it invalid
        return False

def construct_redirect_url(main_url: str, endpoint_name: str, params_dict: dict = None) -> str:
    """
    Constructs a full API redirect URL, validating the base URL's domain and endpoint.

    Args:
        main_url: The base API URL (e.g., "https://api.myservice.com").
        endpoint_name: The specific API endpoint (e.g., "/v1/data").
        params_dict: An optional dictionary of query parameters.

    Returns:
        A complete API redirect URL string.

    Raises:
        ValueError: If the main_url's domain is not in APPROVED_API_DOMAINS
                    or if the endpoint_name is not in ALLOWED_ENDPOINTS.
    """
    try:
        parsed_main_url = urlparse(main_url)
        domain = parsed_main_url.netloc
        if not domain: # Handle cases where main_url might be a relative path or malformed
            raise ValueError(f"Invalid main_url format: '{main_url}'. Could not determine domain.")
    except Exception as e:
        # Catch specific parsing errors or re-raise if needed, but for now, broad catch.
        raise ValueError(f"Invalid main_url format: '{main_url}'. Error: {e}")

    if domain not in APPROVED_API_DOMAINS:
        raise ValueError(f"Domain '{domain}' from main_url '{main_url}' is not an approved API domain.")

    if endpoint_name not in ALLOWED_ENDPOINTS:
        raise ValueError(f"Endpoint '{endpoint_name}' is not an allowed endpoint.")

    # Construct the URL
    # Ensure main_url does not end with '/' if endpoint_name starts with '/', or vice-versa, to avoid '//'
    # A common pattern is to ensure base ends with '/' and endpoint does not start with '/', or use urljoin.
    # For simplicity here: base.rstrip('/') + endpoint
    
    url_path_part = main_url.rstrip('/') + endpoint_name
    
    final_url = url_path_part
    if params_dict:
        query_string = urlencode(params_dict)
        final_url += f"?{query_string}"
    
    return final_url

if __name__ == '__main__':
    print("--- Testing validate_url_and_endpoint (with updated ALLOWED_ENDPOINTS) ---")
    # Note: ALLOWED_ENDPOINTS is now ['/v1/data', '/v1/user', '/v2/analytics', '/health']
    # Original APPROVED_DOMAINS = {'example.com', 'test.com', 'myservice.com'} is still used by this function for domain check.
    
    validation_test_cases = [
        # Valid domain (example.com), but endpoint '/home' is no longer in the new ALLOWED_ENDPOINTS list.
        # The 'endpoint' argument to validate_url_and_endpoint is '/home'.
        # The function first checks if '/home' is in ALLOWED_ENDPOINTS. It's not. So, False.
        ("http://example.com/home", "/home", False),
        # Valid domain (test.com), but endpoint '/about' is no longer in ALLOWED_ENDPOINTS. False.
        ("https://www.test.com/about?query=param", "/about", False),
        # Valid domain (example.com), endpoint '/v1/data' IS in new ALLOWED_ENDPOINTS.
        # URL path '/v1/data' matches endpoint argument '/v1/data'. True.
        ("http://example.com/v1/data", "/v1/data", True),
        # Invalid domain (unknown.com). False.
        ("http://unknown.com/v1/data", "/v1/data", False),
        # Valid domain (example.com), but endpoint argument '/other' is not in ALLOWED_ENDPOINTS. False.
        ("http://example.com/v1/data", "/other", False), 
        # Not a URL. Parsing will fail. False.
        ("not_a_url", "/v1/data", False),
        # Valid domain, endpoint '/v1/user' is allowed. URL path matches. True.
        ("http://myservice.com/v1/user?id=1", "/v1/user", True),
        # Valid domain, endpoint '/v1/user' is allowed. URL path has trailing slash, endpoint arg does not. Mismatch. False.
        ("http://myservice.com/v1/user/?id=1", "/v1/user", False),
    ]

    print("Testing validate_url_and_endpoint:")
    for t_url, t_endpoint, expected_validity in validation_test_cases:
        is_valid = validate_url_and_endpoint(t_url, t_endpoint)
        print(f"  URL: '{t_url}', Endpoint: '{t_endpoint}', Expected: {expected_validity}, Got: {is_valid} -> {'Pass' if is_valid == expected_validity else 'Fail'}")

    print("\n--- Testing construct_redirect_url ---")
    # Uses:
    # APPROVED_API_DOMAINS = {'api.myservice.com', 'api-test.myservice.com', 'api-staging.myservice.com'}
    # ALLOWED_ENDPOINTS = ['/v1/data', '/v1/user', '/v2/analytics', '/health'] (the updated list)

    construction_test_cases = [
        # Valid cases
        ("https://api.myservice.com", "/v1/data", {"id": "123", "type": "raw"}, "https://api.myservice.com/v1/data?id=123&type=raw"),
        ("https://api-test.myservice.com/", "/v1/user", None, "https://api-test.myservice.com/v1/user"), # Trailing slash in main_url
        ("https://api-staging.myservice.com", "/health", {}, "https://api-staging.myservice.com/health"), # Empty params
        # Valid case: main_url contains a path
        ("https://api.myservice.com/basepath", "/v1/data", {"key": "val"}, "https://api.myservice.com/basepath/v1/data?key=val"),
        ("https://api.myservice.com/basepath/", "/v1/user", {"id": "x"}, "https://api.myservice.com/basepath/v1/user?id=x"), # Trailing slash in main_url with path

        # Invalid domain
        ("https://api.wrongservice.com", "/v1/data", None, ValueError),
        # Invalid endpoint
        ("https://api.myservice.com", "/v1/wrong_endpoint", None, ValueError),
        # Invalid main_url format (completely unparseable by netloc)
        ("htp://api.myservice.com", "/v1/data", None, ValueError), # Scheme typo might still parse netloc
        ("api.myservice.com", "/v1/data", None, ValueError), # No scheme, urlparse might treat as path
        ("/relative/path", "/v1/data", None, ValueError), # Relative path, no domain
    ]

    print("Testing construct_redirect_url:")
    for main_url, endpoint, params, expected in construction_test_cases:
        print(f"  Input: main_url='{main_url}', endpoint='{endpoint}', params={params}")
        try:
            result_url = construct_redirect_url(main_url, endpoint, params)
            if expected is ValueError:
                print(f"    Expected: ValueError, Got: '{result_url}' -> Fail (ValueError not raised)")
            elif result_url == expected:
                print(f"    Expected: '{expected}', Got: '{result_url}' -> Pass")
            else:
                print(f"    Expected: '{expected}', Got: '{result_url}' -> Fail (Output mismatch)")
        except ValueError as e:
            if expected is ValueError:
                print(f"    Expected: ValueError, Got: ValueError('{e}') -> Pass")
            else:
                print(f"    Expected: '{expected}', Got: ValueError('{e}') -> Fail (Unexpected ValueError)")
        except Exception as e:
            print(f"    Unexpected Exception: {type(e).__name__}('{e}') -> Fail")
