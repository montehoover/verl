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
    parsed_url = urllib.parse.urlparse(base_url_api)
    domain = parsed_url.netloc
    
    if domain not in APPROVED_API_DOMAINS:
        raise ValueError(f"Domain '{domain}' is not an approved API domain.")

    if desired_endpoint not in ALLOWED_ENDPOINTS:
        raise ValueError(f"Endpoint '{desired_endpoint}' is not an allowed endpoint.")

    # Ensure the base_url_api does not end with a slash if the endpoint starts with one
    # and vice-versa to prevent double slashes or missing slashes.
    # However, urllib.parse.urljoin handles this gracefully.
    
    full_url_path = urllib.parse.urljoin(base_url_api, desired_endpoint)
    
    if optional_params:
        query_string = urllib.parse.urlencode(optional_params)
        return f"{full_url_path}?{query_string}"
    else:
        return full_url_path

if __name__ == '__main__':
    # Example Usage from the problem description
    try:
        url1 = create_redirect_url(
            base_url_api="https://api.myservice.com",
            desired_endpoint="/v1/data",
            optional_params={"user": "123"}
        )
        print(f"Example 1 Output: {url1}") # Expected: https://api.myservice.com/v1/data?user=123

        url2 = create_redirect_url(
            base_url_api="https://api-test.myservice.com",
            desired_endpoint="/v2/analytics"
        )
        print(f"Example 2 Output: {url2}") # Expected: https://api-test.myservice.com/v2/analytics

        # Example of an invalid domain
        try:
            create_redirect_url(
                base_url_api="https://api.unapproved.com",
                desired_endpoint="/v1/data"
            )
        except ValueError as e:
            print(f"Example 3 Error (Invalid Domain): {e}")

        # Example of an invalid endpoint
        try:
            create_redirect_url(
                base_url_api="https://api.myservice.com",
                desired_endpoint="/v3/unknown"
            )
        except ValueError as e:
            print(f"Example 4 Error (Invalid Endpoint): {e}")
            
        # Example with base URL having a path component
        url5 = create_redirect_url(
            base_url_api="https://api.myservice.com/api",
            desired_endpoint="/v1/user",
            optional_params={"id": "456", "active": "true"}
        )
        # urljoin will correctly handle this: https://api.myservice.com/v1/user?id=456&active=true
        # If base_url_api was "https://api.myservice.com/api/" and endpoint was "v1/user" (relative)
        # urljoin would produce "https://api.myservice.com/api/v1/user"
        # If base_url_api was "https://api.myservice.com/api" and endpoint was "/v1/user" (absolute path)
        # urljoin would produce "https://api.myservice.com/v1/user"
        # The current implementation of ALLOWED_ENDPOINTS assumes absolute paths from the domain root.
        # If base_url_api itself contains path components that should be preserved before the desired_endpoint,
        # and desired_endpoint is relative to that path, the logic might need adjustment or clarification.
        # Given ALLOWED_ENDPOINTS are absolute paths, urljoin(base_url_api, desired_endpoint) is robust.
        print(f"Example 5 Output (Base URL with path): {url5}")


    except ValueError as e:
        print(f"An unexpected error occurred during example runs: {e}")
